"""
Shared Memory Control Channel

Dedicated communication channel for control commands between decision and vehicle modules.

Memory Layout:
    [Header: 48 bytes][Control Data: 32 bytes] = Total: 80 bytes

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
from multiprocessing import shared_memory, Lock

from lkas.detection.integration.messages import ControlMessage, ControlMode


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

    Memory layout (32 bytes):
    - steering: 8 bytes (double) - Range [-1.0, 1.0]
    - throttle: 8 bytes (double) - Range [0.0, 1.0]
    - brake: 8 bytes (double) - Range [0.0, 1.0]
    - lateral_offset: 8 bytes (double) - Normalized lateral offset (for telemetry)

    Note: heading_angle omitted to keep structure compact. Can add in future if needed.
    """
    steering: float
    throttle: float
    brake: float
    lateral_offset: float

    @staticmethod
    def byte_size():
        """Size in bytes: 4 doubles = 32 bytes"""
        return struct.calcsize('dddd')

    def pack(self) -> bytes:
        """Pack control data to bytes."""
        return struct.pack(
            'dddd',
            self.steering,
            self.throttle,
            self.brake,
            self.lateral_offset
        )

    @staticmethod
    def unpack(data: bytes) -> 'SharedControlData':
        """Unpack control data from bytes."""
        values = struct.unpack('dddd', data)
        return SharedControlData(
            steering=values[0],
            throttle=values[1],
            brake=values[2],
            lateral_offset=values[3]
        )

    def to_control_message(self, frame_id: int, timestamp: float, mode: ControlMode) -> ControlMessage:
        """Convert to ControlMessage."""
        return ControlMessage(
            steering=self.steering,
            throttle=self.throttle,
            brake=self.brake,
            mode=mode,
            lateral_offset=self.lateral_offset,
            heading_angle=None  # Not stored in shared memory (space optimization)
        )

    @staticmethod
    def from_control_message(control: ControlMessage) -> 'SharedControlData':
        """Create from ControlMessage."""
        return SharedControlData(
            steering=control.steering,
            throttle=control.throttle,
            brake=control.brake,
            lateral_offset=control.lateral_offset or 0.0
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
        [Header: 48 bytes][Control Data: 32 bytes] = 80 bytes total

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
        """Close shared memory."""
        # Release memory views first
        del self.header_view
        del self.data_view
        self.shm.close()

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
