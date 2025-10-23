"""
Test CARLA Connection
Simple script to verify CARLA server is accessible and working.
Run this after starting CARLA Docker container.
"""

import sys
import time


def test_carla_import():
    """Test if carla module can be imported."""
    print("=" * 60)
    print("Step 1: Testing CARLA module import")
    print("=" * 60)

    try:
        import carla

        print("✅ CARLA module imported successfully!")

        # Try to get version if available
        try:
            if hasattr(carla, "__version__"):
                print(f"   Version: {carla.__version__}")
        except:
            pass

        return True
    except ImportError as e:
        print("❌ Failed to import CARLA module")
        print(f"   Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install CARLA: pip install carla==0.9.15")
        print("2. Or set PYTHONPATH to CARLA PythonAPI directory")
        print("   export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla")
        return False


def test_carla_connection(host="carla-server.local", port=2000, timeout=10):
    """Test connection to CARLA server."""
    print("\n" + "=" * 60)
    print("Step 2: Testing CARLA server connection")
    print("=" * 60)
    print(f"Connecting to {host}:{port}...")

    try:
        import carla

        # Create client
        client = carla.Client(host, port)
        client.set_timeout(timeout)

        # Try to get world
        world = client.get_world()

        print("✅ Connected to CARLA server successfully!")
        print(f"   Server version: {client.get_server_version()}")
        print(f"   World map: {world.get_map().name}")

        return True, client

    except RuntimeError as e:
        print("❌ Failed to connect to CARLA server")
        print(f"   Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure CARLA Docker container is running:")
        print("   docker run --rm --platform linux/amd64 -p 2000-2002:2000-2002 \\")
        print("     carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen")
        print("\n2. Check if container is running:")
        print("   docker ps")
        print("\n3. Check Docker logs:")
        print("   docker logs <container-id>")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None


def test_spawn_vehicle(client):
    """Test spawning a vehicle."""
    print("\n" + "=" * 60)
    print("Step 3: Testing vehicle spawn")
    print("=" * 60)

    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Get a vehicle blueprint
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            print("❌ No spawn points available")
            return False, None

        # Spawn vehicle
        print(f"Spawning vehicle at spawn point 0...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

        print("✅ Vehicle spawned successfully!")
        print(f"   Vehicle ID: {vehicle.id}")
        print(f"   Location: {vehicle.get_location()}")

        return True, vehicle

    except Exception as e:
        print(f"❌ Failed to spawn vehicle: {e}")
        return False, None


def test_camera_attach(client, vehicle):
    """Test attaching a camera sensor."""
    print("\n" + "=" * 60)
    print("Step 4: Testing camera sensor")
    print("=" * 60)

    try:
        import carla

        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Get camera blueprint
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")

        # Set camera transform
        camera_transform = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.5), carla.Rotation(pitch=-10.0)
        )

        # Spawn camera
        print("Attaching camera to vehicle...")
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        print("✅ Camera attached successfully!")
        print(f"   Camera ID: {camera.id}")

        # Wait a moment for camera to initialize
        time.sleep(1)

        return True, camera

    except Exception as e:
        print(f"❌ Failed to attach camera: {e}")
        return False, None


def cleanup(vehicle=None, camera=None):
    """Clean up spawned actors."""
    print("\n" + "=" * 60)
    print("Cleanup")
    print("=" * 60)

    try:
        if camera:
            camera.destroy()
            print("✅ Camera destroyed")

        if vehicle:
            vehicle.destroy()
            print("✅ Vehicle destroyed")

    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run all tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test CARLA connection")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="CARLA server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=2000, help="CARLA server port (default: 2000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Connection timeout in seconds (default: 10)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("CARLA Connection Test")
    print("=" * 60)
    print(f"\nTarget: {args.host}:{args.port}")
    print("\nThis script will test:")
    print("1. CARLA module import")
    print("2. Connection to CARLA server")
    print("3. Vehicle spawning")
    print("4. Camera sensor attachment")
    print("\nMake sure CARLA server is running on the target machine!")
    print("=" * 60 + "\n")

    vehicle = None
    camera = None

    try:
        # Test 1: Import
        if not test_carla_import():
            sys.exit(1)

        # Test 2: Connection
        success, client = test_carla_connection(args.host, args.port, args.timeout)
        if not success:
            sys.exit(1)

        # Test 3: Spawn vehicle
        success, vehicle = test_spawn_vehicle(client)
        if not success:
            cleanup(vehicle)
            sys.exit(1)

        # Test 4: Attach camera
        success, camera = test_camera_attach(client, vehicle)
        if not success:
            cleanup(vehicle, camera)
            sys.exit(1)

        # Success!
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour CARLA setup is working correctly!")
        print("You can now run the lane detection system:")
        print("  python main.py --method cv")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Always cleanup
        cleanup(vehicle, camera)


if __name__ == "__main__":
    main()
