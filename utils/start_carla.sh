#!/bin/bash
# CARLA Startup Script for macOS M1/M2
# This script starts CARLA in Docker with optimal settings for Apple Silicon

set -e

echo "================================"
echo "CARLA Simulator Startup (Docker)"
echo "================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    echo ""
    echo "Opening Docker Desktop..."
    open -a Docker
    echo "Waiting for Docker to start..."
    sleep 10
fi

# Check if Docker is ready
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker failed to start. Please start Docker Desktop manually."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if CARLA image exists
echo "Checking for CARLA image..."
if ! docker image inspect carlasim/carla:0.9.15 > /dev/null 2>&1; then
    echo "⚠️  CARLA image not found locally."
    echo "Pulling CARLA 0.9.15 image (this may take a while, ~6-8 GB)..."
    docker pull carlasim/carla:0.9.15
fi

echo "✅ CARLA image ready"
echo ""

# Quality setting
QUALITY="${1:-Low}"

echo "Starting CARLA server..."
echo "  Quality: $QUALITY"
echo "  Platform: linux/amd64 (Rosetta 2 emulation)"
echo "  Ports: 2000-2002"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

# Start CARLA
docker run \
    --rm \
    --name carla-server \
    --platform linux/amd64 \
    -p 2000:2000 \
    -p 2001:2001 \
    -p 2002:2002 \
    carlasim/carla:0.9.15 \
    ./CarlaUE4.sh -RenderOffScreen -quality-level=$QUALITY -benchmark -fps=20

echo ""
echo "CARLA server stopped."
