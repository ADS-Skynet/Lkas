# Distributed Lane Keeping System 🚀

## Overview

The system now supports **two deployment modes**:

| Mode | Architecture | Use Case |
|------|-------------|----------|
| **Local** | Single process | Development, testing, simple deployments |
| **Distributed** | Multi-process | Production, ML models, GPU servers, scalability |

## 🎯 Distributed Architecture (NEW!)

```
┌─────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED SYSTEM                         │
└─────────────────────────────────────────────────────────────┘

PROCESS 1: CARLA Client          PROCESS 2: Detection Server
┌──────────────────────┐          ┌──────────────────────┐
│  Terminal 1          │   ZMQ    │  Terminal 2          │
│                      │◄────────►│                      │
│  main_distributed.py │  Socket  │  detection_server.py │
│                      │          │                      │
│  - CARLA connection  │          │  - Load DL model     │
│  - Vehicle control   │          │  - GPU processing    │
│  - Decision making   │          │  - Return lanes      │
│  - 30 FPS loop       │          │  - Optimized ML      │
└──────────────────────┘          └──────────────────────┘
   Can be anywhere                  Can be on GPU server
```

## Quick Start

### Option 1: Local Mode (Single Process)
```bash
# Simple, all-in-one
python main_modular.py --method cv
```

### Option 2: Distributed Mode (Multi-Process) ⭐ NEW!
```bash
# Terminal 1: Start detection server
python detection_server.py --method cv --port 5555

# Terminal 2: Start CARLA client
python main_distributed.py --detector-url tcp://localhost:5555
```

### Option 3: Remote GPU Server
```bash
# On GPU server (192.168.1.100)
python detection_server.py --method dl --port 5555 --gpu 0

# On CARLA machine
python main_distributed.py --detector-url tcp://192.168.1.100:5555
```

## Architecture Comparison

### Local Mode (`main_modular.py`)

```
┌──────────────────────────────┐
│  Single Python Process       │
│                              │
│  ┌────────┐  ┌──────────┐  │
│  │ CARLA  │─►│Detection │  │
│  │        │◄─│          │  │
│  └────────┘  └──────────┘  │
│       │                     │
│       ▼                     │
│  ┌────────┐                │
│  │Decision│                │
│  └────────┘                │
└──────────────────────────────┘
```

**Pros:**
- ✅ Simple setup
- ✅ Easy debugging
- ✅ Good for prototyping

**Cons:**
- ❌ Detection blocks control loop
- ❌ Can't use remote GPU
- ❌ All or nothing (crash = everything stops)

### Distributed Mode (`main_distributed.py`) ⭐

```
┌─────────────────┐      ┌─────────────────┐
│ CARLA Client    │─────►│ Detection       │
│ Process         │ ZMQ  │ Server Process  │
│                 │◄─────│                 │
│ - Control loop  │      │ - ML inference  │
│ - Fast (30 FPS) │      │ - GPU optimized │
│ - Lightweight   │      │ - Can restart   │
└─────────────────┘      └─────────────────┘
```

**Pros:**
- ✅ **ML Ready**: Detection on GPU, control on CPU
- ✅ **Fault Isolation**: Detection crash doesn't kill vehicle
- ✅ **Scalability**: Multiple clients, load balancing
- ✅ **Flexibility**: Can run on different machines
- ✅ **Hot Reload**: Update model without restarting CARLA

**Cons:**
- ⚠️ Slightly more complex setup (2 terminals)
- ⚠️ Network latency (typically <10ms on localhost)

## Command Reference

### Detection Server

```bash
# Start server with Computer Vision
python detection_server.py --method cv --port 5555

# Start server with Deep Learning
python detection_server.py --method dl --port 5555

# Start on specific GPU
python detection_server.py --method dl --port 5555 --gpu 0

# Bind to all network interfaces (allow remote connections)
python detection_server.py --method cv --port 5555 --host 0.0.0.0

# Use custom config
python detection_server.py --method cv --config my_config.yaml
```

### CARLA Client (Distributed)

```bash
# Connect to local detection server
python main_distributed.py --detector-url tcp://localhost:5555

# Connect to remote detection server
python main_distributed.py --detector-url tcp://192.168.1.100:5555

# Customize CARLA connection
python main_distributed.py \
    --host localhost \
    --port 2000 \
    --spawn-point 5 \
    --detector-url tcp://localhost:5555

# Increase timeout for slow models
python main_distributed.py \
    --detector-url tcp://localhost:5555 \
    --detector-timeout 2000  # 2 seconds

# No visualization (headless)
python main_distributed.py --no-display --detector-url tcp://localhost:5555
```

## Communication Protocol

### ZMQ (ZeroMQ)

We use ZMQ for inter-process communication because:
- ⚡ **Fast**: Near-native socket performance
- 🎯 **Simple**: No broker needed
- 🔧 **Flexible**: TCP, IPC, inproc transports
- 🔄 **Reliable**: Built-in reconnection

### Message Format

**Request (CARLA → Detection):**
```
[Metadata Length: 4 bytes]
[Metadata: JSON]
{
  "timestamp": 123.456,
  "frame_id": 42,
  "width": 800,
  "height": 600
}
[Image Data: JPEG compressed]
```

**Response (Detection → CARLA):**
```json
{
  "frame_id": 42,
  "timestamp": 123.456,
  "processing_time_ms": 15.3,
  "left_lane": {
    "x1": 200, "y1": 600,
    "x2": 350, "y2": 400,
    "confidence": 0.95
  },
  "right_lane": {
    "x1": 600, "y1": 600,
    "x2": 450, "y2": 400,
    "confidence": 0.92
  }
}
```

### Optimizations

1. **JPEG Compression**: Images compressed to ~10% original size
2. **No Debug Images**: Visualization done locally, not sent over network
3. **Async Ready**: Can be easily converted to async/await
4. **Timeout Handling**: Graceful degradation if detection fails

## Deployment Scenarios

### Scenario 1: Local Development
```bash
# Both on same machine, simple debugging
Terminal 1: python detection_server.py --method cv --port 5555
Terminal 2: python main_distributed.py --detector-url tcp://localhost:5555
```

### Scenario 2: GPU Server
```bash
# Detection on powerful GPU server, CARLA on another machine
GPU Server:    python detection_server.py --method dl --gpu 0 --host 0.0.0.0 --port 5555
CARLA Machine: python main_distributed.py --detector-url tcp://192.168.1.100:5555
```

### Scenario 3: Docker Deployment
```bash
# Detection in Docker container with GPU support
docker run --gpus all -p 5555:5555 lane-detection-server
python main_distributed.py --detector-url tcp://localhost:5555
```

### Scenario 4: Multiple Cameras
```bash
# One detection server, multiple CARLA clients
Terminal 1: python detection_server.py --method dl --port 5555
Terminal 2: python main_distributed.py --detector-url tcp://localhost:5555 --spawn-point 0
Terminal 3: python main_distributed.py --detector-url tcp://localhost:5555 --spawn-point 5
```

### Scenario 5: Cloud ML Service
```bash
# Future: Detection on cloud (AWS SageMaker, etc.)
python main_distributed.py --detector-url tcp://ml-server.example.com:5555
```

## Performance

### Latency Breakdown

| Component | Typical Time |
|-----------|-------------|
| Image capture | 1-2ms |
| JPEG compression | 5-10ms |
| Network send (local) | <1ms |
| Detection processing | 10-50ms (CV) / 20-100ms (DL) |
| Network receive | <1ms |
| Control computation | <1ms |
| **Total** | **20-160ms** |

### Network Bandwidth

| Data | Size |
|------|------|
| Raw image (800x600 RGB) | 1.4 MB |
| JPEG compressed | ~50-150 KB |
| Detection response | ~500 bytes |

At 30 FPS: ~1.5-4.5 MB/s (easily handled by gigabit LAN)

## Error Handling

The system gracefully handles:

1. **Detection Timeout**: Applies safe brake, continues with next frame
2. **Network Error**: Reconnects automatically
3. **Server Crash**: Client continues, retries connection
4. **Slow Detection**: Client tracks and reports timeouts

## Monitoring

### Client-Side Metrics
```
Frame 150 | FPS: 28.5 | Lanes: LR | Network: 45.2ms | Steering: -0.123 | Timeouts: 0
```

### Server-Side Metrics
```
Processed 150 frames | Last: 42.1ms | Lanes: LR
```

## Future Enhancements

With distributed architecture, these become easy:

1. **Load Balancing**: Route requests to multiple GPU servers
2. **A/B Testing**: Compare different models in parallel
3. **Monitoring**: Centralized metrics collection
4. **Caching**: Cache results for similar frames
5. **Batch Processing**: Group frames for better GPU utilization
6. **Cloud Integration**: AWS, GCP, Azure ML services
7. **ROS Integration**: Bridge to ROS ecosystem

## Migration Guide

### From Single Process to Distributed

**Before:**
```bash
python main_modular.py --method cv
```

**After:**
```bash
# Terminal 1
python detection_server.py --method cv --port 5555

# Terminal 2
python main_distributed.py --detector-url tcp://localhost:5555
```

**Code Changes:** None! Same configuration, same models.

## Troubleshooting

### "Failed to connect to detection server"
```bash
# Make sure server is running first
python detection_server.py --port 5555

# Then start client
python main_distributed.py --detector-url tcp://localhost:5555
```

### High network latency
```bash
# Increase timeout
python main_distributed.py --detector-timeout 2000

# Or check network connection to remote server
ping 192.168.1.100
```

### Detection timeouts
```bash
# Check server logs in Terminal 1
# If model is slow, increase timeout or use faster model
python main_distributed.py --detector-timeout 2000
```

## Summary

| Feature | Local Mode | Distributed Mode |
|---------|-----------|-----------------|
| Setup | Simple | Two terminals |
| Performance | Good | Excellent |
| ML Support | Basic | Production-ready |
| Fault Tolerance | None | High |
| Scalability | Limited | High |
| Remote GPU | No | Yes |
| **Use When** | **Prototyping** | **Production** |

**Recommendation**: Start with local mode for development, move to distributed mode for production and ML deployment! 🚀
