"""
LKAS ZMQ Broker

Main broker that runs in the LKAS process and manages all ZMQ communication:
- Receives parameter updates from viewer, forwards to detection/decision servers
- Receives action requests from viewer, routes to action handlers
- Receives vehicle status from simulation
- Broadcasts frames, detection, and vehicle status to viewers
"""

import zmq
import json
import time
import numpy as np
from typing import Callable, Dict, Optional, Any

from .broadcaster import VehicleBroadcaster


class LKASBroker:
    """
    Main LKAS broker that manages ZMQ routing and broadcasting.

    Runs in the LKAS main process and provides:
    1. Parameter routing: viewer → broker → detection/decision servers
    2. Action routing: viewer → broker → simulation
    3. Vehicle status routing: simulation → broker → viewers
    4. Broadcasting: frames/detection/state → viewers

    Data flow:
    - LKAS reads detection/frames from shared memory → broadcasts to viewers
    - Simulation sends vehicle status → LKAS broker → broadcasts to viewers
    - Viewer sends parameters → LKAS broker → forwards to servers
    - Viewer sends actions → LKAS broker → forwards to simulation
    """

    def __init__(
        self,
        # Parameter broker URLs
        parameter_viewer_url: str = "tcp://*:5559",  # Receive from viewer
        parameter_servers_url: str = "tcp://*:5560",  # Forward to servers

        # Action URLs
        action_url: str = "tcp://*:5558",  # Receive from viewer
        action_forward_url: str = "tcp://*:5561",  # Forward to simulation

        # Vehicle status URL (receive from simulation)
        vehicle_status_url: str = "tcp://*:5562",  # Receive from simulation

        # Broadcast URL (send to viewers)
        broadcast_url: str = "tcp://*:5557",  # Broadcast to viewers

        # Optional shared ZMQ context
        context: zmq.Context | None = None,

        # Verbose logging
        verbose: bool = False,
    ):
        """
        Initialize LKAS broker.

        Args:
            parameter_viewer_url: URL to receive parameter updates from viewer
            parameter_servers_url: URL to forward parameter updates to servers
            action_url: URL to receive action requests from viewer
            action_forward_url: URL to forward action requests to simulation
            vehicle_status_url: URL to receive vehicle status from simulation
            broadcast_url: URL to broadcast frames/detection/state to viewers
            context: Shared ZMQ context (optional)
        """
        print("\n" + "=" * 60)
        print("LKAS ZMQ Broker")
        print("=" * 60)

        # Create or use provided context
        self.context = context if context else zmq.Context()
        self.owns_context = context is None

        # =====================================================================
        # 1. Parameter Broker: Viewer → Broker → Servers
        # =====================================================================

        # Subscribe to parameter updates from viewer
        self.param_sub_socket = self.context.socket(zmq.SUB)
        self.param_sub_socket.bind(parameter_viewer_url)
        self.param_sub_socket.setsockopt(zmq.SUBSCRIBE, b'parameter')  # Subscribe to 'parameter' topic
        self.param_sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        print(f"✓ Parameter subscriber: {parameter_viewer_url}")

        # Publish parameter updates to detection/decision servers
        self.param_pub_socket = self.context.socket(zmq.PUB)
        self.param_pub_socket.bind(parameter_servers_url)
        print(f"✓ Parameter publisher: {parameter_servers_url}")

        # =====================================================================
        # 2. Action Router: Viewer → Broker → Handlers
        # =====================================================================

        # Subscribe to action requests from viewer
        self.action_socket = self.context.socket(zmq.SUB)
        self.action_socket.bind(action_url)
        self.action_socket.setsockopt(zmq.SUBSCRIBE, b'action')  # Subscribe to 'action' topic
        self.action_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        print(f"✓ Action subscriber: {action_url}")

        # Publish actions to simulation (forward from viewer)
        self.action_pub_socket = self.context.socket(zmq.PUB)
        self.action_pub_socket.bind(action_forward_url)
        print(f"✓ Action publisher (to simulation): {action_forward_url}")

        # Action callbacks: {action_name: callback}
        self.action_callbacks: Dict[str, Callable] = {}

        # =====================================================================
        # 3. Vehicle Status: Simulation → Broker
        # =====================================================================

        # Subscribe to vehicle status from simulation
        self.vehicle_status_socket = self.context.socket(zmq.SUB)
        self.vehicle_status_socket.bind(vehicle_status_url)
        self.vehicle_status_socket.setsockopt(zmq.SUBSCRIBE, b'vehicle_status')
        self.vehicle_status_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        print(f"✓ Vehicle status subscriber: {vehicle_status_url}")

        # =====================================================================
        # 4. Broadcaster: Broker → Viewers
        # =====================================================================

        # Create broadcaster for frames, detection, and vehicle status
        self.broadcaster = VehicleBroadcaster(bind_url=broadcast_url, context=self.context)
        print(f"✓ Broadcaster initialized: {broadcast_url}")

        # Stats
        self.param_forward_count = 0
        self.action_count = 0
        self.vehicle_status_count = 0

        # Verbose logging
        self.verbose = verbose

        print("=" * 60)
        print("Broker ready to route and broadcast")
        print("=" * 60 + "\n")

    # =========================================================================
    # Parameter Routing
    # =========================================================================

    def _poll_parameters(self) -> bool:
        """
        Poll for parameter updates from viewer (non-blocking).

        Returns:
            True if a message was received, False otherwise
        """
        try:
            # Receive from viewer: [topic, json_data]
            parts = self.param_sub_socket.recv_multipart(zmq.NOBLOCK)

            if len(parts) < 2:
                print(f"[Broker] Invalid parameter message: {len(parts)} parts")
                return False

            topic = parts[0].decode('utf-8')
            json_str = parts[1].decode('utf-8')

            # Check if JSON is empty
            if not json_str or json_str.strip() == '':
                print(f"[Broker] Empty parameter JSON received")
                return False

            data = json.loads(json_str)

            # Extract parameter info
            category = data['category']  # 'detection' or 'decision'
            parameter = data['parameter']
            value = data['value']

            # Forward to detection/decision servers
            # Use category as topic so servers can filter
            self.param_pub_socket.send_multipart([
                category.encode('utf-8'),  # Topic = category
                json.dumps(data).encode('utf-8')
            ])

            self.param_forward_count += 1

            # Log parameter forwards for debugging
            if self.verbose:
                print(f"[Broker] Parameter forwarded: {category}.{parameter} = {value}")

            return True

        except zmq.Again:
            # No message available (normal)
            return False
        except json.JSONDecodeError as e:
            print(f"[Broker] JSON decode error: {e}")
            return False
        except Exception as e:
            print(f"[Broker] Error forwarding parameter: {e}")
            return False

    # =========================================================================
    # Action Routing
    # =========================================================================

    def register_action(self, action: str, callback: Callable):
        """
        Register callback for action requests.

        Args:
            action: Action name (e.g., 'respawn', 'pause', 'resume')
            callback: Function to call when action is received
                      Signature: callback(**params)

        Example:
            broker.register_action('pause', on_pause)
            broker.register_action('resume', on_resume)
        """
        self.action_callbacks[action] = callback
        print(f"[Broker] Registered action: {action}")

    def _poll_actions(self) -> bool:
        """
        Poll for action requests from viewer (non-blocking).

        Returns:
            True if a message was received, False otherwise
        """
        try:
            # Receive from viewer: [topic, json_data]
            parts = self.action_socket.recv_multipart(zmq.NOBLOCK)

            topic = parts[0].decode('utf-8')
            data = json.loads(parts[1].decode('utf-8'))

            # Extract action info
            action = data['action']
            params = data.get('params', {})

            if self.verbose:
                print(f"[Broker] Action received: {action}")

            # Forward action to simulation (for pause/resume/respawn)
            self.action_pub_socket.send_multipart([
                b'action',
                json.dumps(data).encode('utf-8')
            ], flags=zmq.NOBLOCK)

            if self.verbose:
                print(f"[Broker] Action forwarded to vehicle: {action}")

            # Also route to registered local handlers (if any)
            if action in self.action_callbacks:
                try:
                    self.action_callbacks[action](**params)
                except Exception as e:
                    print(f"[Broker] Error executing action '{action}': {e}")

            self.action_count += 1
            return True

        except zmq.Again:
            # No message available (normal)
            return False
        except Exception as e:
            print(f"[Broker] Error processing action: {e}")
            return False

    # =========================================================================
    # Vehicle Status Routing
    # =========================================================================

    def _poll_vehicle_status(self) -> bool:
        """
        Poll for vehicle status updates from simulation (non-blocking).

        Returns:
            True if a message was received, False otherwise
        """
        try:
            # Receive from simulation: [topic, json_data]
            parts = self.vehicle_status_socket.recv_multipart(zmq.NOBLOCK)

            if len(parts) < 2:
                print(f"[Broker] Invalid vehicle status message: {len(parts)} parts")
                return False

            topic = parts[0].decode('utf-8')
            data = json.loads(parts[1].decode('utf-8'))

            # Forward simulation's VehicleState directly to viewers
            # Simulation sends: steering, throttle, brake, speed_kmh, position, rotation, paused
            # Viewer expects the SAME format (from simulation.integration.zmq_broadcast.VehicleState)
            # So we just forward it as-is with the 'state' topic

            self.broadcaster.socket.send_multipart([
                b'state',
                json.dumps(data).encode('utf-8')
            ])

            self.vehicle_status_count += 1

            # Debug: Log pause state changes (disabled - use --verbose flag on lkas launcher)
            if self.verbose and self.vehicle_status_count % 50 == 0:  # Every 50 messages
                paused = data.get('paused', False)
                steering = data.get('steering', 0.0)
                speed_kmh = data.get('speed_kmh', 0.0)
                print(f"[Broker] Vehicle status: paused={paused}, steering={steering:.3f}, speed={speed_kmh:.1f}km/h")

            return True

        except zmq.Again:
            # No message available (normal)
            return False
        except Exception as e:
            print(f"[Broker] Error processing vehicle status: {e}")
            import traceback
            traceback.print_exc()
            return False

    # =========================================================================
    # Broadcasting Methods
    # =========================================================================

    def broadcast_frame(self, image: np.ndarray, frame_id: int, jpeg_quality: int = 85, raw_rgb: bool = False):
        """
        Broadcast frame to viewers.

        Args:
            image: Image array (RGB or BGR)
            frame_id: Frame sequence number
            jpeg_quality: JPEG compression quality (0-100)
            raw_rgb: Send raw RGB instead of JPEG (faster for localhost)
        """
        self.broadcaster.send_frame(image, frame_id, jpeg_quality, raw_rgb)

    def broadcast_detection(self, detection_data: Dict[str, Any], frame_id: int = None):
        """
        Broadcast detection data to viewers.

        Args:
            detection_data: Detection data (left_lane, right_lane, etc.)
            frame_id: Frame sequence number (optional)
        """
        self.broadcaster.send_detection(detection_data, frame_id)

    # =========================================================================
    # Main Loop Integration
    # =========================================================================

    def poll(self):
        """
        Poll for all incoming messages (non-blocking).

        Call this regularly in your main loop to process:
        - Parameter updates from viewer
        - Action requests from viewer
        - Vehicle status from simulation

        Example:
            while running:
                broker.poll()
                # ... do LKAS work ...
                # Broadcast frame and detection
                broker.broadcast_frame(image, frame_id)
                broker.broadcast_detection(detection_data, frame_id)
        """
        self._poll_parameters()
        self._poll_actions()
        self._poll_vehicle_status()

    # =========================================================================
    # Cleanup
    # =========================================================================

    def close(self):
        """Close broker and cleanup resources."""
        print("\n[Broker] Shutting down...")

        # Close broadcaster
        if self.broadcaster:
            self.broadcaster.close()

        # Close sockets
        if self.param_sub_socket:
            self.param_sub_socket.close()
        if self.param_pub_socket:
            self.param_pub_socket.close()
        if self.action_socket:
            self.action_socket.close()
        if self.action_pub_socket:
            self.action_pub_socket.close()
        if self.vehicle_status_socket:
            self.vehicle_status_socket.close()

        # Terminate context if we own it
        if self.owns_context and self.context:
            self.context.term()

        print("[Broker] Shutdown complete")

    def get_stats(self) -> Dict[str, int]:
        """Get broker statistics."""
        return {
            'parameters_forwarded': self.param_forward_count,
            'actions_received': self.action_count,
            'vehicle_status_received': self.vehicle_status_count,
        }


# =============================================================================
# Usage Example
# =============================================================================

"""
# LKAS Main Process (run.py)
# ===========================

from lkas.integration.zmq import LKASBroker

# Initialize broker with --broadcast flag
if args.broadcast:
    broker = LKASBroker()

    # Register action handlers
    broker.register_action('pause', on_pause)
    broker.register_action('resume', on_resume)
    broker.register_action('respawn', on_respawn)
else:
    broker = None

# Main loop
while running:
    # Poll for incoming messages (non-blocking)
    # This handles:
    # - Parameter updates from viewer (forwarded to detection/decision servers)
    # - Action requests from viewer (forwarded to simulation + local handlers)
    # - Vehicle status from simulation (forwarded to viewers in original format)
    if broker:
        broker.poll()

    # Read subprocess outputs
    read_detection_output()
    read_decision_output()

    # Broadcast frames and detection data
    if broker:
        broker.broadcast_frame(image, frame_id)
        broker.broadcast_detection(detection_data, frame_id)

# Cleanup
if broker:
    broker.close()
"""


def create_broker_from_config(verbose: bool = False) -> LKASBroker:
    """
    Create LKASBroker with settings from common config.

    This factory function loads ZMQ ports from the shared configuration,
    ensuring consistency across all modules.

    Args:
        verbose: Enable verbose logging

    Returns:
        LKASBroker instance configured from common config
    """
    try:
        from skynet_common.config import ConfigManager
        config = ConfigManager.load()
        comm = config.communication

        # Build URLs from common config
        # Note: These are standard LKAS broker ports based on common config
        # Parameter broker: receive from viewer (zmq_parameter_port), forward to servers
        parameter_viewer_url = f"tcp://*:{comm.zmq_parameter_port}"
        parameter_servers_url = f"tcp://*:{comm.zmq_parameter_port + 1}"  # 5560

        # Action broker: receive from viewer (zmq_action_port), forward to simulation
        action_url = f"tcp://*:{comm.zmq_action_port}"
        action_forward_url = f"tcp://*:{comm.zmq_action_port + 3}"  # 5561

        # Vehicle status: receive from simulation
        vehicle_status_url = f"tcp://*:{comm.zmq_action_port + 4}"  # 5562

        # Broadcast: send to viewers
        broadcast_url = f"tcp://*:{comm.zmq_broadcast_port}"

        return LKASBroker(
            parameter_viewer_url=parameter_viewer_url,
            parameter_servers_url=parameter_servers_url,
            action_url=action_url,
            action_forward_url=action_forward_url,
            vehicle_status_url=vehicle_status_url,
            broadcast_url=broadcast_url,
            verbose=verbose
        )
    except Exception as e:
        print(f"Warning: Could not load common config: {e}")
        print("Using default broker settings")
        return LKASBroker(verbose=verbose)
