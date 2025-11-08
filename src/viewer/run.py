"""
ZMQ-based Web Viewer

Separate process that:
1. Receives data from vehicle via ZMQ (frames, detections, state)
2. Draws overlays on laptop (offloads vehicle CPU)
3. Serves web interface for browser viewing
4. Sends commands back to vehicle via ZMQ

This replaces the threaded web viewer with a proper process-based architecture.
"""

import numpy as np
import cv2
import zmq
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import ZMQ communication and visualization tools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.integration.zmq_broadcast import ViewerSubscriber, ActionPublisher, DetectionData, VehicleState, ParameterPublisher
from simulation.utils.visualizer import LKASVisualizer
from lkas.detection.core.models import LaneDepartureStatus


class ZMQWebViewer:
    """
    Web viewer that receives vehicle data via ZMQ and draws overlays.

    Runs on laptop. Vehicle CPU stays free!
    """

    def __init__(self,
                 vehicle_url: str = "tcp://localhost:5557",
                 action_url: str = "tcp://localhost:5558",
                 parameter_bind_url: str = "tcp://*:5559",
                 web_port: int = 8080,
                 verbose: bool = False,
                 lkas_mode: bool = True):
        """
        Initialize ZMQ web viewer.

        Args:
            vehicle_url: ZMQ URL to receive data from vehicle
            action_url: ZMQ URL to send actions to vehicle
            parameter_bind_url: ZMQ URL to bind/connect for parameter updates
            web_port: HTTP port for web interface
            verbose: Enable verbose HTTP request logging
            lkas_mode: If True, connect to LKAS broker (default, new architecture).
                      If False, bind as server for simulation (old architecture).
        """
        self.vehicle_url = vehicle_url
        self.action_url = action_url
        self.parameter_bind_url = parameter_bind_url
        self.web_port = web_port
        self.verbose = verbose
        self.lkas_mode = lkas_mode

        # ZMQ communication
        self.subscriber = ViewerSubscriber(vehicle_url)
        self.action_publisher = ActionPublisher(action_url)
        self.parameter_publisher = ParameterPublisher(
            bind_url=parameter_bind_url,
            connect_mode=lkas_mode  # Connect to LKAS broker in new architecture
        )

        # Visualization
        self.visualizer = LKASVisualizer()

        # Latest data from vehicle
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_detection: Optional[DetectionData] = None
        self.latest_state: Optional[VehicleState] = None
        self.latest_metrics: Dict[str, Any] = {}

        # Rendered frame with overlays (drawn on laptop!)
        self.rendered_frame: Optional[np.ndarray] = None
        self.render_lock = Thread()

        # HTTP server
        self.http_server: Optional[HTTPServer] = None
        self.http_thread: Optional[Thread] = None
        self.running = False

        print(f"\n{'='*60}")
        print("ZMQ Web Viewer - Laptop Side")
        print(f"{'='*60}")
        print(f"  Receiving from: {vehicle_url}")
        print(f"  Sending actions to: {action_url}")
        print(f"  Parameter server: {parameter_bind_url} ({'connect' if lkas_mode else 'bind'} mode)")
        print(f"  Web interface: http://localhost:{web_port}")
        print(f"{'='*60}\n")

    def start(self):
        """Start viewer (ZMQ polling + HTTP server)."""
        self.running = True

        # Register ZMQ callbacks
        self.subscriber.register_frame_callback(self._on_frame_received)
        self.subscriber.register_detection_callback(self._on_detection_received)
        self.subscriber.register_state_callback(self._on_state_received)

        # Start HTTP server
        self._start_http_server()

        # Start ZMQ polling thread
        zmq_thread = Thread(target=self._zmq_poll_loop, daemon=True)
        zmq_thread.start()

        print("✓ ZMQ Web Viewer started")
        print(f"  Open: http://localhost:{self.web_port}")
        print("  Press Ctrl+C to stop\n")

    def _on_frame_received(self, image: np.ndarray, metadata: Dict):
        """Called when new frame received from vehicle."""
        self.latest_frame = image

        # Render frame with overlays (on laptop, not vehicle!)
        self._render_frame()

    def _on_detection_received(self, detection: DetectionData):
        """Called when detection results received."""
        self.latest_detection = detection
        self._render_frame()

    def _on_state_received(self, state: VehicleState):
        """Called when vehicle state received."""
        self.latest_state = state
        self._render_frame()

    def _render_frame(self):
        """
        Render frame with overlays.

        THIS RUNS ON LAPTOP, NOT VEHICLE!
        Heavy drawing operations don't impact vehicle performance.
        """
        if self.latest_frame is None:
            return

        # Start with original frame
        output = self.latest_frame.copy()

        # Draw lane overlays if detection available
        if self.latest_detection:
            left_lane = None
            right_lane = None

            if self.latest_detection.left_lane:
                ll = self.latest_detection.left_lane
                left_lane = (int(ll['x1']), int(ll['y1']), int(ll['x2']), int(ll['y2']))

            if self.latest_detection.right_lane:
                rl = self.latest_detection.right_lane
                right_lane = (int(rl['x1']), int(rl['y1']), int(rl['x2']), int(rl['y2']))

            # Draw lanes
            output = self.visualizer.draw_lanes(output, left_lane, right_lane, fill_lane=True)

        # Draw vehicle state overlay if available
        if self.latest_state:
            vehicle_telemetry = {
                'speed_kmh': self.latest_state.speed_kmh,
                'position': self.latest_state.position,
                'rotation': self.latest_state.rotation
            }

            # Create metrics dict for HUD
            metrics = {
                'departure_status': LaneDepartureStatus.CENTERED,  # TODO: Calculate from lanes
                'lateral_offset_meters': None,  # TODO: Calculate
                'heading_angle_deg': None,  # TODO: Calculate
                'lane_width_pixels': None  # TODO: Calculate
            }

            # Draw HUD with vehicle data
            output = self.visualizer.draw_hud(
                output,
                metrics,
                show_steering=True,
                steering_value=self.latest_state.steering,
                vehicle_telemetry=vehicle_telemetry
            )

            # Add performance info
            if self.latest_detection:
                cv2.putText(
                    output,
                    f"Detection: {self.latest_detection.processing_time_ms:.1f}ms",
                    (10, output.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # Store rendered frame
        self.rendered_frame = output

    def _zmq_poll_loop(self):
        """ZMQ polling loop (runs in separate thread)."""
        print("[ZMQ] Polling loop started")

        while self.running:
            # Poll for new messages
            self.subscriber.poll()
            time.sleep(0.001)  # Small sleep to prevent busy-wait

        print("[ZMQ] Polling loop stopped")

    def _start_http_server(self):
        """Start HTTP server for web interface."""
        viewer_self = self

        class ViewerRequestHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Custom logging based on verbose flag
                if viewer_self.verbose:
                    # Verbose mode: log all requests with details
                    message = format % args
                    print(f"[HTTP] {message}")
                else:
                    # Normal mode: log important messages only
                    message = format % args
                    if "code 404" in message or "code 500" in message or "error" in message.lower():
                        print(f"[HTTP] {message}")
                    # Suppress routine logs (200, 204)

            def log_request(self, code='-', size='-'):
                """Log an HTTP request with detailed information in verbose mode."""
                if viewer_self.verbose:
                    # Extract client info
                    client_ip = self.client_address[0]
                    client_port = self.client_address[1]

                    # Get request details
                    method = self.command
                    path = self.path
                    protocol = self.request_version

                    # Format log message
                    print(f"[HTTP] {method} {path} {protocol} - Client: {client_ip}:{client_port} - Status: {code} - Size: {size}")
                else:
                    # Use default logging (will be filtered by log_message)
                    super().log_request(code, size)

            def do_POST(self):
                """Handle POST requests for actions and parameters."""
                if self.path == '/action':
                    try:
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))
                        action = data.get('action')

                        # Send action to vehicle via ZMQ
                        # The viewer will update its footer when it receives the state update from simulation
                        viewer_self.action_publisher.send_action(action)

                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'ok', 'action': action})
                        self.wfile.write(response.encode())

                    except Exception as e:
                        print(f"[Action] Error: {e}")
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'error', 'message': str(e)})
                        self.wfile.write(response.encode())

                elif self.path == '/parameter':
                    try:
                        content_length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(content_length)
                        data = json.loads(post_data.decode('utf-8'))

                        category = data.get('category')  # 'detection' or 'decision'
                        parameter = data.get('parameter')
                        value = float(data.get('value'))

                        # Send parameter update via ZMQ
                        viewer_self.parameter_publisher.send_parameter(category, parameter, value)

                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({
                            'status': 'ok',
                            'category': category,
                            'parameter': parameter,
                            'value': value
                        })
                        self.wfile.write(response.encode())

                    except Exception as e:
                        print(f"[Parameter] Error: {e}")
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        response = json.dumps({'status': 'error', 'message': str(e)})
                        self.wfile.write(response.encode())

                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == '/':
                    # Serve HTML page
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    html = self._get_html()
                    self.wfile.write(html.encode())

                elif self.path == '/stream':
                    # Serve MJPEG stream
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.end_headers()

                    frame_count = 0
                    try:
                        while viewer_self.running:
                            if viewer_self.rendered_frame is not None:
                                frame_count += 1

                                # Encode frame as JPEG
                                success, buffer = cv2.imencode('.jpg', viewer_self.rendered_frame,
                                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                                if success:
                                    frame_bytes = buffer.tobytes()

                                    # Send frame
                                    self.wfile.write(b'--jpgboundary\r\n')
                                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                                    self.wfile.write(f'Content-Length: {len(frame_bytes)}\r\n\r\n'.encode())
                                    self.wfile.write(frame_bytes)
                                    self.wfile.write(b'\r\n')
                                else:
                                    print(f"[HTTP] Failed to encode frame!")
                            else:
                                if frame_count == 0:
                                    time.sleep(1)
                                    continue

                            time.sleep(0.033)  # ~30 FPS
                    except Exception as e:
                        print(f"[HTTP] Stream ended: {e}")

                elif self.path == '/status':
                    # Status endpoint - returns current pause state
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    status = {
                        'paused': viewer_self.subscriber.paused,
                        'state_received': viewer_self.subscriber.state_received
                    }
                    self.wfile.write(json.dumps(status).encode())

                elif self.path == '/health':
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK\n')

                elif self.path == '/favicon.ico':
                    self.send_response(204)
                    self.end_headers()
                else:
                    print(f"[HTTP] 404 - Path not found: {self.path}")
                    self.send_error(404)

            def _get_html(self):
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Lane Detection Viewer (ZMQ)</title>
                    <style>
                        body {{
                            margin: 0;
                            padding: 20px;
                            background: #1e1e1e;
                            color: #ffffff;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }}
                        h1 {{
                            margin: 0 0 20px 0;
                            font-size: 24px;
                            font-weight: 300;
                        }}
                        .container {{
                            max-width: 1200px;
                            width: 100%;
                        }}
                        .video-container {{
                            position: relative;
                            width: 100%;
                            background: #000;
                            border-radius: 8px;
                            overflow: hidden;
                            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                        }}
                        img {{
                            width: 100%;
                            height: auto;
                            display: block;
                        }}
                        .controls {{
                            margin-top: 20px;
                            padding: 15px;
                            background: #2d2d2d;
                            border-radius: 8px;
                            display: flex;
                            gap: 10px;
                            flex-wrap: wrap;
                        }}
                        .btn {{
                            padding: 10px 20px;
                            background: #4a4a4a;
                            border: none;
                            border-radius: 4px;
                            color: #fff;
                            cursor: pointer;
                            font-size: 14px;
                            transition: background 0.2s;
                        }}
                        .btn:hover {{
                            background: #5a5a5a;
                        }}
                        .btn.primary {{
                            background: #2196F3;
                        }}
                        .btn.primary:hover {{
                            background: #1976D2;
                        }}
                        .btn.warning {{
                            background: #FF9800;
                        }}
                        .btn.warning:hover {{
                            background: #F57C00;
                        }}
                        .info {{
                            margin-top: 20px;
                            padding: 15px;
                            background: #2d2d2d;
                            border-radius: 8px;
                            font-size: 14px;
                        }}
                        .info-item {{
                            margin: 5px 0;
                        }}
                        .badge {{
                            display: inline-block;
                            padding: 2px 8px;
                            background: #4CAF50;
                            border-radius: 3px;
                            font-size: 12px;
                            font-weight: bold;
                        }}
                        .notification {{
                            position: fixed;
                            top: 20px;
                            right: 20px;
                            background: #4CAF50;
                            color: white;
                            padding: 15px 20px;
                            border-radius: 4px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                            opacity: 0;
                            transition: opacity 0.3s;
                            z-index: 1000;
                        }}
                        .notification.show {{
                            opacity: 1;
                        }}
                        .notification.error {{
                            background: #f44336;
                        }}
                        .params-container {{
                            margin-top: 20px;
                            display: grid;
                            grid-template-columns: 1fr 1fr;
                            gap: 20px;
                        }}
                        .param-section {{
                            padding: 15px;
                            background: #2d2d2d;
                            border-radius: 8px;
                        }}
                        .param-section h3 {{
                            margin: 0 0 15px 0;
                            font-size: 16px;
                            color: #2196F3;
                        }}
                        .param-control {{
                            margin-bottom: 12px;
                        }}
                        .param-control label {{
                            display: block;
                            margin-bottom: 5px;
                            font-size: 13px;
                            color: #aaa;
                        }}
                        .param-control input[type="range"] {{
                            width: 100%;
                            margin-bottom: 5px;
                        }}
                        .param-value {{
                            font-size: 12px;
                            color: #4CAF50;
                            font-family: monospace;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🚗 Lane Detection Viewer <span class="badge">ZMQ MODE</span></h1>
                        <div class="video-container">
                            <img src="/stream" alt="Lane Detection Stream">
                        </div>
                        <div class="controls">
                            <button class="btn primary" onclick="sendAction('respawn')">🔄 Respawn Vehicle</button>
                            <button class="btn warning" id="pauseBtn" onclick="togglePause()">⏸ Pause</button>
                        </div>
                        <div class="info">
                            <div class="info-item">
                                <strong>Mode:</strong> ZMQ-based (Process Separated)
                            </div>
                            <div class="info-item">
                                <strong>Vehicle URL:</strong> {viewer_self.vehicle_url}
                            </div>
                            <div class="info-item">
                                <strong>Benefits:</strong> Overlays rendered on laptop, vehicle CPU stays free!
                            </div>
                        </div>

                        <div class="params-container">
                            <div class="param-section">
                                <h3>🔍 Detection Parameters</h3>
                                <div class="param-control">
                                    <label>Canny Low Threshold</label>
                                    <input type="range" id="canny_low" min="1" max="150" value="50" step="1"
                                           oninput="updateParam('detection', 'canny_low', this.value)">
                                    <span class="param-value" id="canny_low_val">50</span>
                                </div>
                                <div class="param-control">
                                    <label>Canny High Threshold</label>
                                    <input type="range" id="canny_high" min="50" max="255" value="150" step="1"
                                           oninput="updateParam('detection', 'canny_high', this.value)">
                                    <span class="param-value" id="canny_high_val">150</span>
                                </div>
                                <div class="param-control">
                                    <label>Hough Threshold</label>
                                    <input type="range" id="hough_threshold" min="1" max="150" value="50" step="1"
                                           oninput="updateParam('detection', 'hough_threshold', this.value)">
                                    <span class="param-value" id="hough_threshold_val">50</span>
                                </div>
                                <div class="param-control">
                                    <label>Hough Min Line Length</label>
                                    <input type="range" id="hough_min_line_len" min="10" max="150" value="40" step="5"
                                           oninput="updateParam('detection', 'hough_min_line_len', this.value)">
                                    <span class="param-value" id="hough_min_line_len_val">40</span>
                                </div>
                                <div class="param-control">
                                    <label>Smoothing Factor</label>
                                    <input type="range" id="smoothing_factor" min="0" max="1" value="0.7" step="0.05"
                                           oninput="updateParam('detection', 'smoothing_factor', this.value)">
                                    <span class="param-value" id="smoothing_factor_val">0.70</span>
                                </div>
                            </div>

                            <div class="param-section">
                                <h3>🎯 Decision (PID Control) Parameters</h3>
                                <div class="param-control">
                                    <label>Kp (Proportional Gain)</label>
                                    <input type="range" id="kp" min="0" max="2" value="0.5" step="0.05"
                                           oninput="updateParam('decision', 'kp', this.value)">
                                    <span class="param-value" id="kp_val">0.50</span>
                                </div>
                                <div class="param-control">
                                    <label>Kd (Derivative Gain)</label>
                                    <input type="range" id="kd" min="0" max="1" value="0.1" step="0.05"
                                           oninput="updateParam('decision', 'kd', this.value)">
                                    <span class="param-value" id="kd_val">0.10</span>
                                </div>
                                <div class="param-control">
                                    <label>Base Throttle</label>
                                    <input type="range" id="throttle_base" min="0" max="0.5" value="0.14" step="0.01"
                                           oninput="updateParam('decision', 'throttle_base', this.value)">
                                    <span class="param-value" id="throttle_base_val">0.14</span>
                                </div>
                                <div class="param-control">
                                    <label>Min Throttle</label>
                                    <input type="range" id="throttle_min" min="0" max="0.2" value="0.05" step="0.01"
                                           oninput="updateParam('decision', 'throttle_min', this.value)">
                                    <span class="param-value" id="throttle_min_val">0.05</span>
                                </div>
                                <div class="param-control">
                                    <label>Steer Threshold</label>
                                    <input type="range" id="steer_threshold" min="0" max="0.5" value="0.15" step="0.01"
                                           oninput="updateParam('decision', 'steer_threshold', this.value)">
                                    <span class="param-value" id="steer_threshold_val">0.15</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="notification" id="notification"></div>

                    <script>
                        let isPaused = false;

                        function updateParam(category, parameter, value) {{
                            // Update display
                            const displayElem = document.getElementById(parameter + '_val');
                            if (displayElem) {{
                                // Format value based on parameter type
                                if (parameter.includes('throttle') || parameter.includes('kp') ||
                                    parameter.includes('kd') || parameter.includes('steer') ||
                                    parameter.includes('smoothing')) {{
                                    displayElem.textContent = parseFloat(value).toFixed(2);
                                }} else {{
                                    displayElem.textContent = Math.round(value);
                                }}
                            }}

                            // Send to server
                            fetch('/parameter', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json',
                                }},
                                body: JSON.stringify({{
                                    category: category,
                                    parameter: parameter,
                                    value: parseFloat(value)
                                }})
                            }})
                            .then(response => response.json())
                            .then(data => {{
                                if (data.status !== 'ok') {{
                                    console.error('Parameter update failed:', data);
                                }}
                            }})
                            .catch(error => {{
                                console.error('Parameter update error:', error);
                            }});
                        }}

                        function updateButtonState(paused) {{
                            isPaused = paused;
                            const btn = document.getElementById('pauseBtn');
                            if (isPaused) {{
                                btn.textContent = '▶️ Resume';
                                btn.classList.add('primary');
                                btn.classList.remove('warning');
                            }} else {{
                                btn.textContent = '⏸ Pause';
                                btn.classList.remove('primary');
                                btn.classList.add('warning');
                            }}
                        }}

                        function checkStatus() {{
                            fetch('/status')
                                .then(response => response.json())
                                .then(data => {{
                                    if (data.state_received) {{
                                        updateButtonState(data.paused);
                                    }}
                                }})
                                .catch(error => {{
                                    console.error('Status check failed:', error);
                                }});
                        }}

                        function sendAction(action) {{
                            fetch('/action', {{
                                method: 'POST',
                                headers: {{
                                    'Content-Type': 'application/json',
                                }},
                                body: JSON.stringify({{ action: action }})
                            }})
                            .then(response => response.json())
                            .then(data => {{
                                if (data.status === 'ok') {{
                                    showNotification('Action executed: ' + action);
                                }} else {{
                                    showNotification('Error: ' + data.message, true);
                                }}
                            }})
                            .catch(error => {{
                                showNotification('Network error: ' + error, true);
                            }});
                        }}

                        function togglePause() {{
                            // Send the opposite action based on current state
                            sendAction(isPaused ? 'resume' : 'pause');
                        }}

                        function showNotification(message, isError = false) {{
                            const notif = document.getElementById('notification');
                            notif.textContent = message;
                            notif.className = 'notification show' + (isError ? ' error' : '');

                            setTimeout(() => {{
                                notif.classList.remove('show');
                            }}, 3000);
                        }}

                        // Keyboard shortcuts
                        document.addEventListener('keydown', function(event) {{
                            if (event.key === 'r' || event.key === 'R') {{
                                event.preventDefault();
                                sendAction('respawn');
                            }} else if (event.key === ' ') {{
                                event.preventDefault();
                                togglePause();
                            }}
                        }});

                        // Check status on page load
                        checkStatus();

                        // Poll status every 2 seconds to keep button in sync
                        setInterval(checkStatus, 2000);
                    </script>
                </body>
                </html>
                """

        # Start HTTP server with error handling wrapper
        def serve_with_error_handling():
            try:
                self.http_server.serve_forever()
            except Exception as e:
                print(f"[HTTP] Server thread crashed: {e}")
                import traceback
                traceback.print_exc()

        try:
            self.http_server = ThreadingHTTPServer(('0.0.0.0', self.web_port), ViewerRequestHandler)
            self.http_thread = Thread(target=serve_with_error_handling, daemon=True)
            self.http_thread.start()

            # Give server a moment to start
            time.sleep(0.2)

            # Verify thread is still running
            if self.http_thread.is_alive():
                print(f"✓ HTTP server started on port {self.web_port}")
                print(f"  Server listening on: http://0.0.0.0:{self.web_port}")
                print(f"  Local access: http://localhost:{self.web_port}")
                print(f"  Remote access (via VSCode): http://localhost:{self.web_port}")
                print(f"")
            else:
                print(f"✗ HTTP server thread died immediately!")
        except Exception as e:
            print(f"✗ Failed to start HTTP server: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop viewer."""
        self.running = False

        if self.http_server:
            self.http_server.shutdown()

        self.subscriber.close()
        self.action_publisher.close()
        self.parameter_publisher.close()

        print("✓ ZMQ Web Viewer stopped")

    def run(self):
        """Run viewer (blocks until Ctrl+C)."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping viewer...")
            self.stop()


def main():
    """Main entry point for ZMQ web viewer."""
    import argparse
    from lkas.detection.core.config import ConfigManager

    parser = argparse.ArgumentParser(description="ZMQ Web Viewer - Laptop Side")
    parser.add_argument('--config', type=str, default=None,
                       help="Path to configuration file (default: <project-root>/config.yaml)")
    parser.add_argument('--vehicle', type=str, default="tcp://localhost:5557",
                       help="ZMQ URL to receive vehicle data")
    parser.add_argument('--actions', type=str, default="tcp://localhost:5558",
                       help="ZMQ URL to send actions")
    parser.add_argument('--parameters', type=str, default="tcp://localhost:5559",
                       help="ZMQ URL to send parameter updates")
    parser.add_argument('--port', type=int, default=None,
                       help="HTTP port for web interface (overrides config, default: from config.yaml)")
    parser.add_argument('--verbose', action='store_true',
                       help="Enable verbose HTTP request logging")
    parser.add_argument('--simulation-mode', action='store_true',
                       help="Use simulation mode (bind as server). Default is LKAS mode (connect to broker).")

    args = parser.parse_args()

    # Load configuration
    config = ConfigManager.load(args.config)

    # Determine web port: CLI arg overrides config
    web_port = args.port if args.port is not None else config.visualization.web_port

    # Determine mode: LKAS mode (connect to broker) is default
    lkas_mode = not args.simulation_mode

    # Create and run viewer
    viewer = ZMQWebViewer(
        vehicle_url=args.vehicle,
        action_url=args.actions,
        parameter_bind_url=args.parameters,
        web_port=web_port,
        verbose=args.verbose,
        lkas_mode=lkas_mode
    )

    viewer.start()
    viewer.run()


# Main entry point
if __name__ == "__main__":
    main()
