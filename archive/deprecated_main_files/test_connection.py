#!/usr/bin/env python3
"""
Test script to verify the detection server connection.
"""
import sys
import zmq
import json
import time

def test_connection(server_url="tcp://localhost:5555", timeout_ms=2000):
    """Test connection to detection server."""
    print(f"Testing connection to {server_url}...")

    # Create ZMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)

    try:
        # Connect to server
        print(f"  Connecting...")
        socket.connect(server_url)
        print(f"  ✓ Socket connected")

        # Send ping
        print(f"  Sending ping message...")
        ping_msg = json.dumps({"type": "ping"}).encode('utf-8')
        socket.send(ping_msg)
        print(f"  ✓ Ping sent")

        # Wait for pong
        print(f"  Waiting for pong response (timeout: {timeout_ms}ms)...")
        response = socket.recv()
        response_data = json.loads(response.decode('utf-8'))
        print(f"  ✓ Received response: {response_data}")

        if response_data.get("type") == "pong":
            print("\n✓✓✓ Connection test PASSED! Server is responding correctly.")
            return True
        else:
            print(f"\n✗✗✗ Connection test FAILED! Unexpected response: {response_data}")
            return False

    except zmq.Again:
        print(f"\n✗✗✗ Connection test FAILED! Timeout - server not responding")
        return False
    except Exception as e:
        print(f"\n✗✗✗ Connection test FAILED! Error: {e}")
        return False
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
