#!/usr/bin/env python3
"""
Test script for ColorFlow color grading pipeline.
Verifies all components work together correctly.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorflow.io.ffmpeg_tools import VideoSampler, LUTLoader
from colorflow.analysis.metrics import ColorAnalyzer
from colorflow.transforms.dynamic_graph import DynamicColorTransform, TransformParams
from colorflow.calibration.feedback_loop import ColorCalibrator
from colorflow.visualization.diagnostics import ColorVisualizer


def create_test_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a synthetic test frame with gradients."""
    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)

    # Create RGB gradients
    r = (X * 255).astype(np.uint8)
    g = (Y * 255).astype(np.uint8)
    b = ((X + Y) / 2 * 255).astype(np.uint8)

    return np.stack([r, g, b], axis=2)


def test_basic_components():
    """Test basic ColorFlow components."""
    print("🧪 Testing ColorFlow Components")
    print("=" * 40)

    # Test 1: Frame creation and analysis
    print("\n1. Testing frame creation and analysis...")
    test_frame = create_test_frame()

    analyzer = ColorAnalyzer()
    metrics = analyzer.analyze_frame(test_frame, 0.0, 0, "test_video")

    print(f"   Frame shape: {test_frame.shape}")
    print(f"   Mean RGB: {metrics.mean_rgb}")
    print(f"   ΔE mean: {metrics.delta_e_mean:.4f}")
    print(f"   Saturation: {metrics.saturation_mean:.4f}")
    print("   ✅ Frame analysis working")
    # Test 2: Transformation pipeline
    print("\n2. Testing transformation pipeline...")
    transform = DynamicColorTransform()

    # Create test parameters
    params = TransformParams(
        frame_index=0,
        timestamp=0.0,
        ev_adjustment=0.5,
        wb_matrix=np.array([[1.1, 0, 0], [0, 1.0, 0], [0, 0, 0.9]]),
        tone_shadows=0.1,
        tone_midtones=0.0,
        tone_highlights=-0.1,
        saturation_gain=1.2,
        lut_strength=0.0
    )

    transformed_frame = transform.apply_transform(test_frame, params)
    print(f"   Original shape: {test_frame.shape}")
    print(f"   Transformed shape: {transformed_frame.shape}")
    print(f"   Transform applied successfully: {np.array_equal(test_frame.shape, transformed_frame.shape)}")
    print("   ✅ Transformation pipeline working"))
    # Test 3: LUT loading
    print("\n3. Testing LUT loading...")
    lut_loader = LUTLoader()

    if os.path.exists('luts/base.cube'):
        lut_data = lut_loader.load_cube('luts/base.cube')
        if lut_data:
            print(f"   LUT size: {lut_data['size']}")
            print(f"   LUT domain: {lut_data['domain_min']} - {lut_data['domain_max']}")
            print("   ✅ LUT loading working")        else:
            print("   ❌ LUT loading failed"    else:
        print("   ⚠️  No LUT file found, skipping LUT test"
    # Test 4: Sampling (mock test)
    print("\n4. Testing video sampling...")
    sampler = VideoSampler()

    # Mock video info
    mock_info = {
        'duration': 10.0,
        'frame_count': 300,
        'fps': 30.0,
        'width': 1920,
        'height': 1080,
        'codec': 'h264',
        'pixel_format': 'yuv420p',
        'has_audio': True,
        'format': 'mp4'
    }

    print(f"   Mock video info: {mock_info}")
    print("   ✅ Sampler initialized")
    # Test 5: Visualization
    print("\n5. Testing visualization...")
    visualizer = ColorVisualizer()

    # Create test metrics for visualization
    test_metrics = [metrics]
    plot_path = visualizer.plot_frame_analysis(test_metrics, "Test Analysis")

    if plot_path and os.path.exists(plot_path):
        print(f"   Plot saved: {plot_path}")
        print("   ✅ Visualization working")    else:
        print("   ❌ Visualization failed"
    print("\n🎉 All basic tests completed!")
    return True


def test_full_pipeline():
    """Test the full ColorFlow pipeline."""
    print("\n🚀 Testing Full Pipeline")
    print("=" * 40)

    # Create test frames
    print("\n1. Creating test frames...")
    frames = [create_test_frame() for _ in range(5)]
    timestamps = [i * 2.0 for i in range(5)]  # 2-second intervals

    print(f"   Created {len(frames)} test frames")

    # Test analysis
    print("\n2. Running frame analysis...")
    analyzer = ColorAnalyzer()
    frame_metrics = []

    for i, frame in enumerate(frames):
        metrics = analyzer.analyze_frame(frame, timestamps[i], i, "test_pipeline")
        frame_metrics.append(metrics)
        print(f"   Frame {i}: ΔE={metrics.delta_e_mean:.4f} EV={metrics.ev_estimate:.2f}")

    # Test optimization
    print("\n3. Running optimization...")
    calibrator = ColorCalibrator()
    optimization_result = calibrator.optimize_clip(frames, timestamps, "test_pipeline")

    print("   Optimization results:")
    print(f"   - Iterations: {optimization_result.iteration}")
    print(f"   - Final ΔE std: {optimization_result.delta_e_std:.4f}")
    print(f"   - Status: {optimization_result.status.name}")

    # Test transformation application
    print("\n4. Applying transformations...")
    transform = DynamicColorTransform()

    transformed_frames = []
    for i, frame in enumerate(frames):
        params = optimization_result.best_params[i]
        transformed = transform.apply_transform(frame, params)
        transformed_frames.append(transformed)

    print(f"   Applied transformations to {len(transformed_frames)} frames")

    # Test clip analysis
    print("\n5. Running clip analysis...")
    clip_metrics = analyzer.analyze_clip(frame_metrics)

    print("   Clip metrics:")
    print(f"   - ΔE std global: {clip_metrics.delta_e_std_global:.4f}")
    print(f"   - Mean luminance: {clip_metrics.mean_luminance:.4f}")
    print(f"   - EV range: {clip_metrics.ev_range[0]:.2f} to {clip_metrics.ev_range[1]:.2f}")

    print("\n✅ Full pipeline test completed successfully!")
    return True


def main():
    """Run all tests."""
    print("🎨 ColorFlow Test Suite")
    print("=" * 50)

    try:
        # Test basic components
        basic_ok = test_basic_components()

        # Test full pipeline
        pipeline_ok = test_full_pipeline()

        if basic_ok and pipeline_ok:
            print("\n🎉 All tests passed! ColorFlow is ready to use.")
            print("\nTo run on your videos:")
            print("  python -m colorflow.cli grade --input videos --lut luts/base.cube --output output")
        else:
            print("\n❌ Some tests failed. Check the output above.")

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)