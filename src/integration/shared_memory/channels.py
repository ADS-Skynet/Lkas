"""
Shared Memory-Based Detection Communication

Ultra-low latency detection server/client using shared memory instead of ZMQ.

Performance:
- Camera → Detection: ~0.001ms (vs ~2ms for ZMQ TCP)
- Detection → Control: ~0.001ms for shared memory, or ZMQ for broadcast

Architecture:
    Camera Process              Detection Process
    ┌─────────────┐            ┌──────────────┐
    │  Write      │────────────►│  Read        │
    │  Image      │ Shared Mem │  Image       │
    │             │  ~0.001ms  │              │
    └─────────────┘            └──────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  Detect      │
                               │  Lanes       │
                               └──────┬───────┘
                                      │
                  ┌───────────────────┴────────────────────┐
                  │                                        │
                  ▼                                        ▼
         ┌─────────────────┐                    ┌──────────────────┐
         │ Write Results   │                    │ ZMQ Broadcast    │
         │ to Shared Mem   │                    │ (for viewers)    │
         │ ~0.001ms        │                    │ ~5ms             │
         └────────┬────────┘                    └──────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Control Process │
         │ Reads Results   │
         └─────────────────┘
"""

import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional
from multiprocessing import shared_memory, Lock, Value, resource_tracker
import struct

from lkas.integration.messages import ImageMessage, DetectionMessage, LaneMessage


# =============================================================================
# Metadata Structures
# =============================================================================

@dataclass
class SharedImageHeader:
    """Header for shared image buffer."""
    frame_id: int
    timestamp: float
    width: int
    height: int
    channels: int
    ready: int  # 1 if new data available, 0 if already consumed

    @staticmethod
    def byte_size():
        """Size in bytes: Calculate actual struct size with padding."""
        return struct.calcsize('qdiiii')

    def pack(self) -> bytes:
        """Pack header to bytes."""
        # Format: q=int64 (frame_id), d=double (timestamp), i=int32 (4 fields: width, height, channels, ready)
        return struct.pack('qdiiii',
                          self.frame_id,
                          self.timestamp,
                          self.width,
                          self.height,
                          self.channels,
                          self.ready)

    @staticmethod
    def unpack(data: bytes) -> 'SharedImageHeader':
        """Unpack header from bytes."""
        values = struct.unpack('qdiiii', data)
        return SharedImageHeader(
            frame_id=values[0],
            timestamp=values[1],
            width=values[2],
            height=values[3],
            channels=values[4],
            ready=values[5]
        )


@dataclass
class SharedDetectionHeader:
    """Header for shared detection results."""
    frame_id: int
    timestamp: float
    processing_time_ms: float
    has_left_lane: int
    has_right_lane: int
    ready: int  # 1 if new data available

    @staticmethod
    def byte_size():
        """Size in bytes: Calculate actual struct size with padding."""
        return struct.calcsize('qddiii')

    def pack(self) -> bytes:
        """Pack header to bytes."""
        return struct.pack('qddiii',
                          self.frame_id,
                          self.timestamp,
                          self.processing_time_ms,
                          self.has_left_lane,
                          self.has_right_lane,
                          self.ready)

    @staticmethod
    def unpack(data: bytes) -> 'SharedDetectionHeader':
        """Unpack header from bytes."""
        values = struct.unpack('qddiii', data)
        return SharedDetectionHeader(
            frame_id=values[0],
            timestamp=values[1],
            processing_time_ms=values[2],
            has_left_lane=values[3],
            has_right_lane=values[4],
            ready=values[5]
        )


@dataclass
class SharedLane:
    """Shared memory representation of a lane."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @staticmethod
    def byte_size():
        """Size in bytes: Calculate actual struct size with padding."""
        return struct.calcsize('iiiid')

    def pack(self) -> bytes:
        """Pack lane to bytes."""
        return struct.pack('iiiid', self.x1, self.y1, self.x2, self.y2, self.confidence)

    @staticmethod
    def unpack(data: bytes) -> 'SharedLane':
        """Unpack lane from bytes."""
        values = struct.unpack('iiiid', data)
        return SharedLane(
            x1=values[0],
            y1=values[1],
            x2=values[2],
            y2=values[3],
            confidence=values[4]
        )

    def to_lane_message(self) -> LaneMessage:
        """Convert to LaneMessage."""
        return LaneMessage(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            confidence=self.confidence
        )

    @staticmethod
    def from_lane_message(lane: LaneMessage) -> 'SharedLane':
        """Create from LaneMessage."""
        return SharedLane(
            x1=int(lane.x1),
            y1=int(lane.y1),
            x2=int(lane.x2),
            y2=int(lane.y2),
            confidence=float(lane.confidence)
        )


# =============================================================================
# Shared Memory Image Writer/Reader (Camera <-> Detection)
# =============================================================================

class SharedMemoryImageChannel:
    """
    High-performance shared memory channel for camera images.

    Memory Layout:
        [Header: 32 bytes][Image Data: width*height*channels bytes]
    """

    def __init__(self, name: str, shape: tuple, create: bool = True, retry_count: int = 10, retry_delay: float = 0.5):
        """
        Initialize shared memory image channel.

        Args:
            name: Unique name for shared memory
            shape: Image shape (height, width, channels)
            create: True to create (writer), False to connect (reader)
            retry_count: Number of retry attempts for connection (default: 10)
            retry_delay: Delay between retries in seconds (default: 0.5)
        """
        self.name = name
        self.shape = shape
        self.height, self.width, self.channels = shape
        self._is_creator = create  # Track if we created this memory
        self._unregistered = False  # Track if we've unregistered from resource tracker

        # Calculate sizes
        self.header_size = SharedImageHeader.byte_size()
        self.image_size = int(np.prod(shape))
        self.total_size = self.header_size + self.image_size

        # Create or connect to shared memory with retry logic
        if create:
            # Cleanup old memory if exists
            try:
                old_shm = shared_memory.SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()
                print(f"  Cleaned up old shared memory: {name}")
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=self.total_size,
                name=name
            )
            print(f"✓ Created shared memory: {name} ({self.total_size} bytes)")
        else:
            # Reader mode: Retry connection if not yet created
            self.shm = None
            for attempt in range(retry_count):
                try:
                    self.shm = shared_memory.SharedMemory(name=name)
                    print(f"\n✓ Connected to shared memory: {name}                    ")
                    break
                except FileNotFoundError:
                    if attempt < retry_count - 1:
                        # print(
                        #     f"  Waiting for shared memory '{name}' "
                        #     f"(attempt {attempt + 1}/{retry_count})...",
                        #     end="\r", flush=True
                        # )
                        time.sleep(retry_delay)
                    else:
                        raise ConnectionError(
                            f"Shared memory '{name}' not found after {retry_count} attempts. "
                            f"Make sure the writer process is running."
                        )

        # Create views
        self.header_view = self.shm.buf[:self.header_size]
        self.image_view = np.ndarray(
            shape,
            dtype=np.uint8,
            buffer=self.shm.buf[self.header_size:]
        )

        # Synchronization
        self.lock = Lock()

    def __del__(self):
        """Destructor - automatically cleanup when object is destroyed."""
        try:
            # Release array views first
            if hasattr(self, 'image_view'):
                del self.image_view
            if hasattr(self, 'header_view'):
                del self.header_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # Creators are handled automatically by unlink()
            # Only unregister if we haven't already done so in close()
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def write(self, image: np.ndarray, timestamp: float, frame_id: int):
        """
        Write image to shared memory.

        Args:
            image: Image array (must match shape)
            timestamp: Image timestamp
            frame_id: Frame sequence number
        """
        if image.shape != self.shape:
            raise ValueError(f"Image shape {image.shape} != expected {self.shape}")

        with self.lock:
            # Write image data (fast memcpy)
            np.copyto(self.image_view, image)

            # Write header
            header = SharedImageHeader(
                frame_id=frame_id,
                timestamp=timestamp,
                width=self.width,
                height=self.height,
                channels=self.channels,
                ready=1
            )
            self.header_view[:] = header.pack()

    def read(self, copy: bool = True) -> Optional[ImageMessage]:
        """
        Read image from shared memory.

        Args:
            copy: If True, returns copy. If False, returns view (faster but unsafe)

        Returns:
            ImageMessage or None if no new data
        """
        with self.lock:
            # Read header
            header = SharedImageHeader.unpack(bytes(self.header_view))

            # Check if data is ready
            if header.ready == 0:
                return None

            # Read image
            if copy:
                image = np.copy(self.image_view)
            else:
                image = self.image_view

            # Mark as consumed (optional - comment out for multiple readers)
            # header.ready = 0
            # self.header_view[:] = header.pack()

            return ImageMessage(
                image=image,
                timestamp=header.timestamp,
                frame_id=header.frame_id
            )

    def read_blocking(self, timeout: float = 1.0, copy: bool = True) -> Optional[ImageMessage]:
        """
        Read image, waiting for new data.

        Args:
            timeout: Maximum wait time in seconds
            copy: Whether to copy image data

        Returns:
            ImageMessage or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.read(copy=copy)
            if result is not None:
                return result
            time.sleep(0.0001)  # 0.1ms sleep
        return None

    def close(self):
        """Close shared memory and unregister from resource tracker."""
        try:
            # Release array views first
            if hasattr(self, 'image_view'):
                del self.image_view
            if hasattr(self, 'header_view'):
                del self.header_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # to prevent "leaked shared_memory" warnings when creator unlinks first
            # Mark as unregistered to prevent double-unregister in __del__
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def unlink(self):
        """Unlink (delete) shared memory."""
        try:
            self.shm.unlink()
            print(f"✓ Cleaned up shared memory: {self.name}")
        except Exception as e:
            print(f"⚠ Error cleaning up shared memory: {e}")


# =============================================================================
# Shared Memory Detection Results Writer/Reader (Detection <-> Control)
# =============================================================================

class SharedMemoryDetectionChannel:
    """
    High-performance shared memory channel for detection results.

    Memory Layout:
        [Header: 40 bytes][Left Lane: 24 bytes][Right Lane: 24 bytes]
    """

    def __init__(self, name: str, create: bool = True, retry_count: int = 10, retry_delay: float = 0.5):
        """
        Initialize shared memory detection channel.

        Args:
            name: Unique name for shared memory
            create: True to create (writer), False to connect (reader)
            retry_count: Number of retry attempts for connection (default: 10)
            retry_delay: Delay between retries in seconds (default: 0.5)
        """
        self.name = name
        self._is_creator = create  # Track if we created this memory
        self._unregistered = False  # Track if we've unregistered from resource tracker

        # Calculate sizes
        self.header_size = SharedDetectionHeader.byte_size()
        self.lane_size = SharedLane.byte_size()
        self.total_size = self.header_size + 2 * self.lane_size  # header + 2 lanes

        # Create or connect to shared memory with retry logic
        if create:
            # Cleanup old memory if exists
            try:
                old_shm = shared_memory.SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()
                print(f"  Cleaned up old detection shared memory: {name}")
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=self.total_size,
                name=name
            )
            print(f"✓ Created detection shared memory: {name} ({self.total_size} bytes)")
        else:
            # Reader mode: Retry connection if not yet created
            self.shm = None
            for attempt in range(retry_count):
                try:
                    self.shm = shared_memory.SharedMemory(name=name)
                    print(f"\r✓ Connected to detection shared memory: {name}                    ")
                    break
                except FileNotFoundError:
                    if attempt < retry_count - 1:
                        # print(
                        #     f"  Waiting for detection shared memory '{name}'"
                        #     f"(attempt {attempt + 1}/{retry_count})...",
                        #     end="\r", flush=True
                        # )
                        time.sleep(retry_delay)
                    else:
                        print(flush=True)  # Clear the retry line
                        raise ConnectionError(
                            f"Detection shared memory '{name}' not found after {retry_count} attempts. "
                            f"Make sure the detection server is running."
                        )

        # Create views
        self.header_view = self.shm.buf[:self.header_size]
        self.left_lane_view = self.shm.buf[self.header_size:self.header_size + self.lane_size]
        self.right_lane_view = self.shm.buf[self.header_size + self.lane_size:self.header_size + 2 * self.lane_size]

        # Synchronization
        self.lock = Lock()

    def __del__(self):
        """Destructor - automatically cleanup when object is destroyed."""
        try:
            # Release memory views first
            if hasattr(self, 'header_view'):
                del self.header_view
            if hasattr(self, 'left_lane_view'):
                del self.left_lane_view
            if hasattr(self, 'right_lane_view'):
                del self.right_lane_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # Creators are handled automatically by unlink()
            # Only unregister if we haven't already done so in close()
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def write(self, detection: DetectionMessage):
        """
        Write detection results to shared memory.

        Args:
            detection: Detection message
        """
        with self.lock:
            # Write header
            header = SharedDetectionHeader(
                frame_id=detection.frame_id,
                timestamp=detection.timestamp,
                processing_time_ms=detection.processing_time_ms,
                has_left_lane=1 if detection.left_lane else 0,
                has_right_lane=1 if detection.right_lane else 0,
                ready=1
            )
            self.header_view[:] = header.pack()

            # Write lanes
            if detection.left_lane:
                left = SharedLane.from_lane_message(detection.left_lane)
                self.left_lane_view[:] = left.pack()

            if detection.right_lane:
                right = SharedLane.from_lane_message(detection.right_lane)
                self.right_lane_view[:] = right.pack()

    def read(self) -> Optional[DetectionMessage]:
        """
        Read detection results from shared memory.

        Returns:
            DetectionMessage or None if no new data
        """
        with self.lock:
            # Read header
            header = SharedDetectionHeader.unpack(bytes(self.header_view))

            # Check if data is ready
            if header.ready == 0:
                return None

            # Read lanes
            left_lane = None
            right_lane = None

            if header.has_left_lane:
                left = SharedLane.unpack(bytes(self.left_lane_view))
                left_lane = left.to_lane_message()

            if header.has_right_lane:
                right = SharedLane.unpack(bytes(self.right_lane_view))
                right_lane = right.to_lane_message()

            # Mark as consumed (optional)
            # header.ready = 0
            # self.header_view[:] = header.pack()

            return DetectionMessage(
                left_lane=left_lane,
                right_lane=right_lane,
                processing_time_ms=header.processing_time_ms,
                frame_id=header.frame_id,
                timestamp=header.timestamp,
                debug_image=None  # Not transmitted via shared memory (too large)
            )

    def close(self):
        """Close shared memory and unregister from resource tracker."""
        try:
            # Release memory views first
            if hasattr(self, 'header_view'):
                del self.header_view
            if hasattr(self, 'left_lane_view'):
                del self.left_lane_view
            if hasattr(self, 'right_lane_view'):
                del self.right_lane_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # to prevent "leaked shared_memory" warnings when creator unlinks first
            # Mark as unregistered to prevent double-unregister in __del__
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def unlink(self):
        """Unlink (delete) shared memory."""
        try:
            self.shm.unlink()
            print(f"✓ Cleaned up detection shared memory: {self.name}")
        except Exception as e:
            print(f"⚠ Error cleaning up detection shared memory: {e}")


# =============================================================================
# Note: DetectionClient has been moved to detection/client.py
# =============================================================================
#
# The DetectionClient class pairs with DetectionServer to form a complete
# client-server architecture using shared memory for communication.
# It has been moved to detection/client.py for better organization.
# =============================================================================
"""
Shared Memory Control Channel

Dedicated communication channel for control commands between decision and vehicle modules.

Memory Layout:
    [Header: 48 bytes][Control Data: 88 bytes] = Total: 136 bytes

Performance:
- Control → Vehicle: ~0.001ms (shared memory copy)
- Minimal overhead: Only 80 bytes per message
- Zero serialization cost

Architecture:
    Decision Process           Vehicle/Simulation Process
    ┌─────────────┐           ┌─────────────┐
    │  Compute    │──────────►│  Read       │
    │  Control    │ Shared    │  Control    │
    │  Commands   │ Memory    │  Commands   │
    └─────────────┘           └─────────────┘
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional
from multiprocessing import shared_memory, Lock, resource_tracker

from lkas.integration.messages import ControlMessage, ControlMode


# =============================================================================
# Control Message Header Structure
# =============================================================================

@dataclass
class SharedControlHeader:
    """
    Header for shared control message.

    Memory layout (48 bytes):
    - frame_id: 8 bytes (int64)
    - timestamp: 8 bytes (double)
    - processing_time_ms: 8 bytes (double)
    - mode: 4 bytes (int32) - ControlMode enum value
    - ready: 4 bytes (int32) - 1 if new data available, 0 if consumed
    - reserved: 16 bytes - for future extensions
    """
    frame_id: int
    timestamp: float
    processing_time_ms: float
    mode: int  # ControlMode as integer
    ready: int  # 1 = new data, 0 = consumed

    @staticmethod
    def byte_size():
        """Size in bytes: q=8, d=8, d=8, i=4, i=4, 16x=16 = 48 bytes"""
        return struct.calcsize('qddii16x')

    def pack(self) -> bytes:
        """Pack header to bytes."""
        return struct.pack(
            'qddii16x',
            self.frame_id,
            self.timestamp,
            self.processing_time_ms,
            self.mode,
            self.ready
        )

    @staticmethod
    def unpack(data: bytes) -> 'SharedControlHeader':
        """Unpack header from bytes."""
        values = struct.unpack('qddii16x', data)
        return SharedControlHeader(
            frame_id=values[0],
            timestamp=values[1],
            processing_time_ms=values[2],
            mode=values[3],
            ready=values[4]
        )


# =============================================================================
# Control Data Structure
# =============================================================================

@dataclass
class SharedControlData:
    """
    Control command data.

    Memory layout (88 bytes):
    - steering: 8 bytes (double) - Range [-1.0, 1.0]
    - throttle: 8 bytes (double) - Range [0.0, 1.0]
    - brake: 8 bytes (double) - Range [0.0, 1.0]
    - lateral_offset: 8 bytes (double) - Normalized lateral offset (for telemetry)
    - lateral_offset_meters: 8 bytes (double) - Lateral offset in meters
    - heading_angle: 8 bytes (double) - Heading angle in degrees
    - lane_width_pixels: 8 bytes (double) - Lane width in pixels
    - departure_status: 32 bytes (string) - Lane departure status string
    """
    steering: float
    throttle: float
    brake: float
    lateral_offset: float
    lateral_offset_meters: float
    heading_angle: float
    lane_width_pixels: float
    departure_status: str

    @staticmethod
    def byte_size():
        """Size in bytes: 7 doubles + 32 char string = 88 bytes"""
        return struct.calcsize('ddddddd32s')

    def pack(self) -> bytes:
        """Pack control data to bytes."""
        # Convert None to 0.0 for numeric fields
        lateral_offset_m = self.lateral_offset_meters if self.lateral_offset_meters is not None else 0.0
        heading = self.heading_angle if self.heading_angle is not None else 0.0
        lane_width = self.lane_width_pixels if self.lane_width_pixels is not None else 0.0

        # Convert status string to bytes (max 32 chars)
        status_bytes = (self.departure_status or '').encode('utf-8')[:31] + b'\x00'
        status_bytes = status_bytes.ljust(32, b'\x00')

        return struct.pack(
            'ddddddd32s',
            self.steering,
            self.throttle,
            self.brake,
            self.lateral_offset,
            lateral_offset_m,
            heading,
            lane_width,
            status_bytes
        )

    @staticmethod
    def unpack(data: bytes) -> 'SharedControlData':
        """Unpack control data from bytes."""
        values = struct.unpack('ddddddd32s', data)

        # Decode status string
        status_str = values[7].rstrip(b'\x00').decode('utf-8') if values[7] else None

        return SharedControlData(
            steering=values[0],
            throttle=values[1],
            brake=values[2],
            lateral_offset=values[3],
            lateral_offset_meters=values[4] if values[4] != 0.0 else None,
            heading_angle=values[5] if values[5] != 0.0 else None,
            lane_width_pixels=values[6] if values[6] != 0.0 else None,
            departure_status=status_str
        )

    def to_control_message(self, frame_id: int, timestamp: float, mode: ControlMode) -> ControlMessage:
        """Convert to ControlMessage."""
        return ControlMessage(
            steering=self.steering,
            throttle=self.throttle,
            brake=self.brake,
            mode=mode,
            lateral_offset=self.lateral_offset,
            lateral_offset_meters=self.lateral_offset_meters,
            heading_angle=self.heading_angle,
            lane_width_pixels=self.lane_width_pixels,
            departure_status=self.departure_status
        )

    @staticmethod
    def from_control_message(control: ControlMessage) -> 'SharedControlData':
        """Create from ControlMessage."""
        return SharedControlData(
            steering=control.steering,
            throttle=control.throttle,
            brake=control.brake,
            lateral_offset=control.lateral_offset or 0.0,
            lateral_offset_meters=control.lateral_offset_meters,
            heading_angle=control.heading_angle,
            lane_width_pixels=control.lane_width_pixels,
            departure_status=control.departure_status
        )


# =============================================================================
# Control Mode Mapping
# =============================================================================

def control_mode_to_int(mode: ControlMode) -> int:
    """Convert ControlMode enum to integer for storage."""
    mode_map = {
        ControlMode.MANUAL: 0,
        ControlMode.AUTOPILOT: 1,
        ControlMode.LANE_KEEPING: 2,
    }
    return mode_map.get(mode, 2)  # Default to LANE_KEEPING


def int_to_control_mode(value: int) -> ControlMode:
    """Convert integer to ControlMode enum."""
    mode_map = {
        0: ControlMode.MANUAL,
        1: ControlMode.AUTOPILOT,
        2: ControlMode.LANE_KEEPING,
    }
    return mode_map.get(value, ControlMode.LANE_KEEPING)


# =============================================================================
# Shared Memory Control Channel
# =============================================================================

class SharedMemoryControlChannel:
    """
    High-performance shared memory channel for control commands.

    Memory Layout:
        [Header: 48 bytes][Control Data: 88 bytes] = 136 bytes total

    Usage:
        # Writer (Decision Process)
        channel = SharedMemoryControlChannel(name="control_commands", create=True)
        channel.write(control_message)

        # Reader (Vehicle Process)
        channel = SharedMemoryControlChannel(name="control_commands", create=False)
        control = channel.read()
    """

    def __init__(
        self,
        name: str,
        create: bool = True,
        retry_count: int = 20,
        retry_delay: float = 0.5
    ):
        """
        Initialize shared memory control channel.

        Args:
            name: Unique name for shared memory
            create: True to create (writer), False to connect (reader)
            retry_count: Number of retry attempts for connection
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self._is_creator = create  # Track if we created this memory
        self._unregistered = False  # Track if we've unregistered from resource tracker

        # Calculate sizes
        self.header_size = SharedControlHeader.byte_size()
        self.data_size = SharedControlData.byte_size()
        self.total_size = self.header_size + self.data_size

        # Create or connect to shared memory
        if create:
            # Cleanup old memory if exists
            try:
                old_shm = shared_memory.SharedMemory(name=name)
                old_shm.close()
                old_shm.unlink()
                print(f"  Cleaned up old control shared memory: {name}")
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                create=True,
                size=self.total_size,
                name=name
            )
            print(f"\n✓ Created control shared memory: {name} ({self.total_size} bytes)")
        else:
            # Reader mode: Retry connection if not yet created
            self.shm = None
            for attempt in range(retry_count):
                try:
                    self.shm = shared_memory.SharedMemory(name=name)
                    print(f"\r✓ Connected to control shared memory: {name}                    ")
                    break
                except FileNotFoundError:
                    if attempt < retry_count - 1:
                        # print(
                        #     f"  Waiting for control shared memory '{name}' "
                        #     f"(attempt {attempt + 1}/{retry_count})...",
                        #     end='\r', flush=True
                        # )
                        time.sleep(retry_delay)
                    else:
                        print(flush=True)  # Clear the retry line
                        raise ConnectionError(
                            f"Control shared memory '{name}' not found after {retry_count} attempts. "
                            f"Make sure the decision server is running."
                        )

        # Create views
        self.header_view = self.shm.buf[:self.header_size]
        self.data_view = self.shm.buf[self.header_size:self.header_size + self.data_size]

        # Synchronization
        self.lock = Lock()

    def __del__(self):
        """Destructor - automatically cleanup when object is destroyed."""
        try:
            # Release memory views first
            if hasattr(self, 'header_view'):
                del self.header_view
            if hasattr(self, 'data_view'):
                del self.data_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # Creators are handled automatically by unlink()
            # Only unregister if we haven't already done so in close()
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def write(self, control: ControlMessage, frame_id: int, timestamp: float, processing_time_ms: float = 0.0):
        """
        Write control message to shared memory.

        Args:
            control: Control message to write
            frame_id: Frame sequence number
            timestamp: Message timestamp
            processing_time_ms: Decision processing time in milliseconds
        """
        with self.lock:
            # Write header
            header = SharedControlHeader(
                frame_id=frame_id,
                timestamp=timestamp,
                processing_time_ms=processing_time_ms,
                mode=control_mode_to_int(control.mode),
                ready=1
            )
            self.header_view[:] = header.pack()

            # Write control data
            data = SharedControlData.from_control_message(control)
            self.data_view[:] = data.pack()

    def read(self) -> Optional[ControlMessage]:
        """
        Read control message from shared memory.

        Returns:
            ControlMessage or None if no new data
        """
        with self.lock:
            # Read header
            header = SharedControlHeader.unpack(bytes(self.header_view))

            # Check if data is ready
            if header.ready == 0:
                return None

            # Read control data
            data = SharedControlData.unpack(bytes(self.data_view))

            # Convert to ControlMessage
            mode = int_to_control_mode(header.mode)
            control = data.to_control_message(
                frame_id=header.frame_id,
                timestamp=header.timestamp,
                mode=mode
            )

            # Mark as consumed (optional - comment out for multiple readers)
            # header.ready = 0
            # self.header_view[:] = header.pack()

            return control

    def read_blocking(self, timeout: float = 1.0) -> Optional[ControlMessage]:
        """
        Read control message, waiting for new data.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            ControlMessage or None if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.read()
            if result is not None:
                return result
            time.sleep(0.0001)  # 0.1ms sleep
        return None

    def close(self):
        """Close shared memory and unregister from resource tracker."""
        try:
            # Release memory views first
            if hasattr(self, 'header_view'):
                del self.header_view
            if hasattr(self, 'data_view'):
                del self.data_view
        except Exception:
            pass

        # Close and unregister shared memory
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
            except Exception:
                pass

            # CRITICAL: Readers must manually unregister from resource tracker
            # to prevent "leaked shared_memory" warnings when creator unlinks first
            # Mark as unregistered to prevent double-unregister in __del__
            if not getattr(self, '_is_creator', False) and not getattr(self, '_unregistered', False):
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                    self._unregistered = True
                except (KeyError, ValueError, AttributeError):
                    pass

    def unlink(self):
        """Unlink (delete) shared memory."""
        try:
            self.shm.unlink()
            print(f"✓ Cleaned up control shared memory: {self.name}")
        except Exception as e:
            print(f"⚠ Error cleaning up control shared memory: {e}")


# =============================================================================
# Protocol Documentation
# =============================================================================

"""
CONTROL MESSAGE PROTOCOL SPECIFICATION
======================================

Version: 1.0
Purpose: Ultra-low latency control command transmission
Size: 80 bytes total (compact and cache-friendly)

MEMORY LAYOUT:
┌────────────────────────────────────────────────────────────┐
│ Header (48 bytes)                                          │
├────────────────────────────────────────────────────────────┤
│ frame_id          │ 8 bytes │ int64  │ Frame sequence     │
│ timestamp         │ 8 bytes │ double │ Message timestamp  │
│ processing_time   │ 8 bytes │ double │ Decision time (ms) │
│ mode              │ 4 bytes │ int32  │ Control mode enum  │
│ ready             │ 4 bytes │ int32  │ Data ready flag    │
│ reserved          │ 16 bytes│ -      │ Future extensions  │
├────────────────────────────────────────────────────────────┤
│ Control Data (32 bytes)                                    │
├────────────────────────────────────────────────────────────┤
│ steering          │ 8 bytes │ double │ [-1.0, 1.0]        │
│ throttle          │ 8 bytes │ double │ [0.0, 1.0]         │
│ brake             │ 8 bytes │ double │ [0.0, 1.0]         │
│ lateral_offset    │ 8 bytes │ double │ Telemetry data     │
└────────────────────────────────────────────────────────────┘

CONTROL MODES (mode field):
- 0: MANUAL         - Manual control (not lane keeping)
- 1: AUTOPILOT      - Full autonomous mode
- 2: LANE_KEEPING   - Lane keeping assist mode (default)
- 3: EMERGENCY_STOP - Emergency stop triggered

READY FLAG (ready field):
- 1: New data available
- 0: Data already consumed

SYNCHRONIZATION:
- Uses multiprocessing.Lock for thread-safe access
- Writer sets ready=1 after writing
- Reader can optionally set ready=0 after reading
- For multiple readers, keep ready=1

PERFORMANCE CHARACTERISTICS:
- Write latency: ~0.001ms (single memcpy)
- Read latency: ~0.001ms (single memcpy)
- Total size: 80 bytes (fits in single cache line on most CPUs)
- No serialization/deserialization overhead
- No network overhead
- No system calls (after initial setup)

USAGE PATTERNS:

Pattern 1: Single Decision → Single Vehicle
    Decision writes, Vehicle reads and consumes (ready=0)

Pattern 2: Single Decision → Multiple Vehicles (broadcast)
    Decision writes, multiple Vehicles read (ready stays 1)

Pattern 3: Telemetry/Logging
    Any process can read without consuming for monitoring

FUTURE EXTENSIONS (using reserved 16 bytes):
- heading_angle: 8 bytes (double) - Vehicle heading angle
- gear: 4 bytes (int32) - Current gear
- turn_signal: 4 bytes (int32) - Turn signal state
- emergency_flags: 4 bytes (bitfield) - Various emergency indicators

EXAMPLE USAGE:

Writer (Decision Server):
    channel = SharedMemoryControlChannel("control_commands", create=True)
    control = ControlMessage(steering=0.3, throttle=0.5, brake=0.0)
    channel.write(control, frame_id=123, timestamp=time.time())

Reader (Vehicle):
    channel = SharedMemoryControlChannel("control_commands", create=False)
    control = channel.read()
    if control:
        vehicle.apply_control(control)
"""

