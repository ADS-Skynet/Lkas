#!/bin/bash

# X11 Setup Script for macOS with XQuartz
# This script helps configure X11 forwarding for the lane detection visualization

echo "=================================================="
echo "X11 Setup for Lane Detection Visualization"
echo "=================================================="
echo ""

# Check if running in container
if [ -f /.dockerenv ]; then
    echo "✓ Running in Docker container"

    # Try to set DISPLAY to host.docker.internal
    export DISPLAY=host.docker.internal:0
    echo "✓ DISPLAY set to: $DISPLAY"

    # Set Qt environment variables
    export QT_X11_NO_MITSHM=1
    export QT_DEBUG_PLUGINS=0

    # Test X11 connection
    echo ""
    echo "Testing X11 connection..."

    if command -v xdpyinfo &> /dev/null; then
        if xdpyinfo -display "$DISPLAY" &> /dev/null; then
            echo "✓ X11 connection successful!"
        else
            echo "✗ X11 connection failed"
            echo ""
            echo "Please ensure on your macOS host:"
            echo "  1. XQuartz is running"
            echo "  2. In XQuartz preferences -> Security:"
            echo "     ☑ Allow connections from network clients"
            echo "  3. Run in macOS terminal: xhost + localhost"
            echo "  4. Restart this container"
        fi
    else
        echo "⚠ xdpyinfo not installed, skipping connection test"
    fi
else
    echo "Not running in container - X11 setup not needed"
fi

echo ""
echo "=================================================="
echo "To run the lane detection with visualization:"
echo "  source setup_x11.sh"
echo "  python3 lane_detection/main.py --host carla-server.local"
echo "=================================================="
