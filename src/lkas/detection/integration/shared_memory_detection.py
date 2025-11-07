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
from multiprocessing import shared_memory, Lock, Value
import struct

from lkas.detection.integration.messages import ImageMessage, DetectionMessage, LaneMessage


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
                        print(flush=True)  # Clear the retry line
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
        """Close shared memory."""
        # Release array views first
        del self.image_view
        del self.header_view
        self.shm.close()

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
        """Close shared memory."""
        # Release memory views first
        del self.header_view
        del self.left_lane_view
        del self.right_lane_view
        self.shm.close()

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
