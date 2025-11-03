#!/usr/bin/env python3
"""
Test script for shared memory communication.

This script tests the shared memory IPC without needing CARLA running.
It simulates a camera sending images and detection receiving them.
"""

import numpy as np
import time
import cv2
from multiprocessing import Process

from simulation.integration.shared_memory_detection import (
    SharedMemoryImageChannel,
    SharedMemoryDetectionChannel,
    SharedDetectionHeader,
    SharedLane,
)
from simulation.integration.messages import DetectionMessage, LaneMessage


def camera_simulator():
    """Simulates camera writing images to shared memory."""
    print("[Camera] Starting camera simulator...")

    # Create shared memory writer
    writer = SharedMemoryImageChannel(
        name="test_camera",
        shape=(480, 640, 3),
        create=True
    )

    print("[Camera] Shared memory created, writing frames...")

    try:
        for frame_id in range(100):
            # Generate test image with frame number
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Add frame number text
            cv2.putText(
                image,
                f"Frame {frame_id}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Write to shared memory
            timestamp = time.time()
            writer.write(image, timestamp=timestamp, frame_id=frame_id)

            if frame_id % 10 == 0:
                print(f"[Camera] Wrote frame {frame_id}")

            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\n[Camera] Stopping...")
    finally:
        writer.close()
        writer.unlink()
        print("[Camera] Cleanup complete")


def detection_simulator():
    """Simulates detection process reading images and writing results."""
    print("[Detection] Starting detection simulator...")
    time.sleep(0.5)  # Wait for camera to create shared memory

    # Connect to image shared memory (reader)
    image_reader = SharedMemoryImageChannel(
        name="test_camera",
        shape=(480, 640, 3),
        create=False
    )

    # Create detection result shared memory (writer)
    detection_writer = SharedMemoryDetectionChannel(
        name="test_detection",
        create=True
    )

    print("[Detection] Connected to shared memory, processing frames...")

    try:
        last_frame_id = -1
        for i in range(100):
            # Read image from shared memory
            image_msg = image_reader.read_blocking(timeout=1.0, copy=True)

            if image_msg is None:
                print("[Detection] Timeout waiting for image")
                continue

            # Skip if same frame
            if image_msg.frame_id == last_frame_id:
                time.sleep(0.001)
                continue

            last_frame_id = image_msg.frame_id

            # Simulate detection processing
            time.sleep(0.015)  # Simulate 15ms processing

            # Create fake detection results
            detection = DetectionMessage(
                left_lane=LaneMessage(x1=100, y1=480, x2=200, y2=240, confidence=0.9),
                right_lane=LaneMessage(x1=540, y1=480, x2=440, y2=240, confidence=0.85),
                processing_time_ms=15.5,
                frame_id=image_msg.frame_id,
                timestamp=time.time()
            )

            # Write detection results to shared memory
            detection_writer.write(detection)

            if image_msg.frame_id % 10 == 0:
                print(f"[Detection] Processed frame {image_msg.frame_id}, "
                      f"shape={image_msg.image.shape}")

    except KeyboardInterrupt:
        print("\n[Detection] Stopping...")
    finally:
        image_reader.close()
        detection_writer.close()
        detection_writer.unlink()
        print("[Detection] Cleanup complete")


def control_simulator():
    """Simulates control process reading detection results."""
    print("[Control] Starting control simulator...")
    time.sleep(1.0)  # Wait for detection to create shared memory

    # Connect to detection shared memory (reader)
    from simulation.integration.shared_memory_detection import SharedMemoryDetectionClient

    detector = SharedMemoryDetectionClient(
        detection_shm_name="test_detection"
    )

    print("[Control] Connected to detection shared memory, reading results...")

    try:
        detection_count = 0
        start_time = time.time()

        for i in range(100):
            # Read detection results
            detection = detector.get_detection(timeout=1.0)

            if detection is None:
                time.sleep(0.001)
                continue

            detection_count += 1

            if detection_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = detection_count / elapsed
                print(f"[Control] Received detection for frame {detection.frame_id}, "
                      f"FPS: {fps:.1f}, "
                      f"left={detection.left_lane is not None}, "
                      f"right={detection.right_lane is not None}")

    except KeyboardInterrupt:
        print("\n[Control] Stopping...")
    finally:
        detector.close()
        print("[Control] Cleanup complete")


def test_latency():
    """Test end-to-end latency of shared memory communication."""
    print("\n" + "=" * 60)
    print("LATENCY TEST")
    print("=" * 60)

    # Setup
    writer = SharedMemoryImageChannel(
        name="latency_test",
        shape=(600, 800, 3),
        create=True
    )

    reader = SharedMemoryImageChannel(
        name="latency_test",
        shape=(600, 800, 3),
        create=False
    )

    # Test write/read latency
    latencies = []

    for i in range(100):
        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        # Write
        start = time.time()
        writer.write(image, timestamp=time.time(), frame_id=i)

        # Read
        result = reader.read(copy=True)
        end = time.time()

        latency_us = (end - start) * 1_000_000  # microseconds
        latencies.append(latency_us)

    # Print statistics
    print(f"\nShared Memory Latency (write + read):")
    print(f"  Samples: {len(latencies)}")
    print(f"  Min: {min(latencies):.2f} μs")
    print(f"  Avg: {sum(latencies) / len(latencies):.2f} μs")
    print(f"  Max: {max(latencies):.2f} μs")
    print(f"  Median: {sorted(latencies)[len(latencies)//2]:.2f} μs")

    # Cleanup
    writer.close()
    writer.unlink()
    reader.close()

    print("\n" + "=" * 60)


def main():
    """Main test entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test shared memory communication")
    parser.add_argument(
        "--test",
        type=str,
        choices=["full", "latency"],
        default="full",
        help="Test type (full=camera+detection+control, latency=latency only)"
    )

    args = parser.parse_args()

    if args.test == "latency":
        test_latency()
    else:
        # Full system test
        print("\n" + "=" * 60)
        print("SHARED MEMORY FULL SYSTEM TEST")
        print("=" * 60)
        print("Testing: Camera → Detection → Control")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Start all processes
        camera_proc = Process(target=camera_simulator)
        detection_proc = Process(target=detection_simulator)
        control_proc = Process(target=control_simulator)

        camera_proc.start()
        detection_proc.start()
        control_proc.start()

        try:
            camera_proc.join()
            detection_proc.join()
            control_proc.join()
        except KeyboardInterrupt:
            print("\n\nStopping all processes...")
            camera_proc.terminate()
            detection_proc.terminate()
            control_proc.terminate()
            camera_proc.join()
            detection_proc.join()
            control_proc.join()

        print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
