"""
Performance Benchmark: Single Process vs Multi-Process Need

This script helps you decide if you need multi-processing by measuring
how long each component takes.

Rule of thumb:
- Total loop time < 33ms → 30 FPS → Single process is fine
- Total loop time < 50ms → 20 FPS → Single process probably OK
- Total loop time > 50ms → <20 FPS → Consider multi-processing
"""

import time
import numpy as np
from core.config import ConfigManager
from core.factory import DetectorFactory


def benchmark_detector(detector_type: str, num_iterations: int = 10):
    """
    Benchmark a detector's performance.

    Args:
        detector_type: 'cv' or 'dl'
        num_iterations: Number of test runs
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {detector_type.upper()} Detector")
    print(f"{'='*70}")

    # Create detector
    config = ConfigManager.load('config.yaml')
    factory = DetectorFactory(config)
    detector = factory.create(detector_type)

    print(f"Detector: {detector.get_name()}")

    # Create test image (simulate CARLA camera)
    test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # Warm-up (important for DL models!)
    print("Warming up...")
    for _ in range(3):
        detector.detect(test_image)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []

    for i in range(num_iterations):
        start = time.time()
        result = detector.detect(test_image)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)

        print(f"  Iteration {i+1:2d}: {elapsed:6.2f}ms")

    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    print(f"\n{'─'*70}")
    print(f"Results for {detector_type.upper()} Detector:")
    print(f"{'─'*70}")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time:     {min_time:.2f}ms")
    print(f"  Max time:     {max_time:.2f}ms")
    print(f"  Std dev:      {std_time:.2f}ms")

    # Calculate FPS
    fps = 1000.0 / avg_time
    print(f"\n  Theoretical FPS: {fps:.1f}")

    # Recommendation
    print(f"\n{'─'*70}")
    print("Recommendation:")
    print(f"{'─'*70}")

    if avg_time < 33:
        print("  ✅ EXCELLENT - Can achieve 30+ FPS")
        print("  → Single process is perfect!")
    elif avg_time < 50:
        print("  ✅ GOOD - Can achieve 20+ FPS")
        print("  → Single process should work fine")
    elif avg_time < 100:
        print("  ⚠️  ACCEPTABLE - Can achieve 10-20 FPS")
        print("  → Single process OK, but consider optimization")
    else:
        print("  ❌ SLOW - Less than 10 FPS")
        print("  → Consider multi-processing or GPU acceleration")

    return avg_time


def estimate_total_loop_time(cv_time: float, dl_time: float):
    """
    Estimate total loop time including all components.

    Args:
        cv_time: CV detector average time (ms)
        dl_time: DL detector average time (ms)
    """
    print(f"\n{'='*70}")
    print("Total Loop Time Estimation")
    print(f"{'='*70}")

    # Typical times for other components (estimated)
    carla_get_image = 5.0   # CARLA sensor read (fast)
    analysis_time = 2.0     # LaneAnalyzer.get_metrics() (fast)
    control_compute = 1.0   # PDController.compute() (very fast)
    visualization = 10.0    # Drawing HUD, overlays (moderate)
    carla_apply = 1.0       # Apply control command (fast)

    overhead = carla_get_image + analysis_time + control_compute + visualization + carla_apply

    print(f"\nComponent breakdown (estimated):")
    print(f"  CARLA get image:     {carla_get_image:6.2f}ms")
    print(f"  Lane detection (CV): {cv_time:6.2f}ms")
    print(f"  Lane detection (DL): {dl_time:6.2f}ms")
    print(f"  Lane analysis:       {analysis_time:6.2f}ms")
    print(f"  Control compute:     {control_compute:6.2f}ms")
    print(f"  Visualization:       {visualization:6.2f}ms")
    print(f"  CARLA apply:         {carla_apply:6.2f}ms")

    # Calculate totals
    cv_total = overhead + cv_time
    dl_total = overhead + dl_time

    print(f"\n{'─'*70}")
    print(f"Total loop time with CV detector: {cv_total:.2f}ms → {1000/cv_total:.1f} FPS")
    print(f"Total loop time with DL detector: {dl_total:.2f}ms → {1000/dl_total:.1f} FPS")
    print(f"{'─'*70}")

    # Recommendations
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}")

    print("\n📊 Based on benchmarks:\n")

    if cv_total < 50:
        print("  ✅ CV Detector: Use SINGLE PROCESS")
        print("     → Fast enough for real-time performance")
        print("     → Simple, easy to debug")
    else:
        print("  ⚠️  CV Detector: Consider MULTI-PROCESS")
        print("     → Run detector in separate process")

    print()

    if dl_total < 50:
        print("  ✅ DL Detector: Use SINGLE PROCESS")
        print("     → Pre-trained model is fast enough")
        print("     → Keep it simple!")
    else:
        print("  ⚠️  DL Detector: Consider MULTI-PROCESS or GPU")
        print("     → CPU inference is slow")
        print("     → Options:")
        print("        1. Run detector in separate process")
        print("        2. Use GPU if available")
        print("        3. Use smaller model (input_size=(128,128))")

    print(f"\n{'='*70}")
    print("ARCHITECTURE RECOMMENDATION")
    print(f"{'='*70}")

    if cv_total < 50 and dl_total < 50:
        print("""
✅ SINGLE PROCESS is recommended:

    class LaneKeepingAssist:
        def run(self):
            while True:
                image = carla.get_image()
                result = detector.detect(image)
                control = controller.compute(result)
                carla.apply(control)

    ✓ Simple architecture
    ✓ Easy to debug
    ✓ Fast enough for real-time
""")
    else:
        print("""
⚠️  MULTI-PROCESS may be beneficial:

    # Detector in separate process
    detector_process = Process(target=detection_worker)
    detector_process.start()

    # Main process
    while True:
        image = carla.get_image()
        image_queue.put(image)
        result = result_queue.get()  # From detector process
        control = controller.compute(result)
        carla.apply(control)

    ✓ Parallelizes slow detection
    ✓ Main loop stays responsive
    ✗ More complex code
    ✗ Harder to debug
""")


def main():
    """Run complete benchmark."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PERFORMANCE BENCHMARK - HELP YOU DECIDE" + " "*14 + "║")
    print("╚" + "="*68 + "╝")

    print("""
This benchmark will help you decide:
    Single Process  vs  Multi-Process

Testing both CV and DL detectors to measure performance...
""")

    # Benchmark CV detector
    cv_time = benchmark_detector('cv', num_iterations=10)

    # Benchmark DL detector
    dl_time = benchmark_detector('dl', num_iterations=10)

    # Estimate total loop time
    estimate_total_loop_time(cv_time, dl_time)

    print(f"\n{'='*70}")
    print("🎯 SUMMARY")
    print(f"{'='*70}")
    print("""
Start with SINGLE PROCESS because:
  ✅ Simpler to develop and debug
  ✅ Easier to understand data flow
  ✅ Your OOP architecture supports both!
  ✅ Can switch to multi-process later if needed

Only use MULTI-PROCESS if:
  ❌ FPS is consistently below 20
  ❌ Detector takes >100ms
  ❌ You need to run CV and DL simultaneously

Your current OOP design makes it easy to switch later!
    """)

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
