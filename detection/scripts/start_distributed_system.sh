#!/bin/bash
# Helper script to start the distributed system
# This opens two terminals with detection server and CARLA client

echo "=========================================="
echo "Starting Distributed Lane Keeping System"
echo "=========================================="
echo ""

# Default values
METHOD=${1:-cv}
PORT=${2:-5555}

echo "Configuration:"
echo "  Detection method: $METHOD"
echo "  Port: $PORT"
echo ""

# Check if we're in the right directory
if [ ! -f "detection_server.py" ]; then
    echo "Error: Please run this script from the lane_detection directory"
    exit 1
fi

echo "Starting detection server in new terminal..."
# Try different terminal emulators
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "python detection_server.py --method $METHOD --port $PORT; exec bash"
elif command -v xterm &> /dev/null; then
    xterm -e "python detection_server.py --method $METHOD --port $PORT; bash" &
elif command -v konsole &> /dev/null; then
    konsole -e "python detection_server.py --method $METHOD --port $PORT; bash" &
else
    echo "Warning: No supported terminal emulator found"
    echo "Please start detection server manually:"
    echo "  python detection_server.py --method $METHOD --port $PORT"
fi

# Wait for server to start
echo "Waiting for detection server to initialize..."
sleep 3

echo ""
echo "Starting CARLA client..."
echo "Press Ctrl+C to stop"
echo ""

python main_distributed.py --detector-url tcp://localhost:$PORT

echo ""
echo "System stopped"
