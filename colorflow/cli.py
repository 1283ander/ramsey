"""
Command-line interface for ColorFlow color grading pipeline.
Provides grade command with comprehensive options for batch processing.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

# Import ColorFlow modules
from .io.ffmpeg_tools import VideoSampler, LUTLoader
from .analysis.metrics import ColorAnalyzer
from .calibration.feedback_loop import ColorCalibrator
from .apply.apply_video import VideoProcessor
from .visualization.diagnostics import ColorVisualizer
from .config import get_config


class ColorFlowCLI:
    """Command-line interface for ColorFlow."""

    def __init__(self):
        """Initialize CLI."""
        self.config = get_config()

    def run(self, args=None):
        """Run CLI with provided arguments."""
        parser = argparse.ArgumentParser(
            description="ColorFlow - Dynamic Color Analysis and Adaptive LUT Pipeline"
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Grade command
        grade_parser = subparsers.add_parser('grade', help='Grade videos with adaptive color correction')
        grade_parser.add_argument('--input', '-i', required=True, help='Input directory with videos')
        grade_parser.add_argument('--lut', '-l', help='Base LUT file (.cube format)')
        grade_parser.add_argument('--output', '-o', default='output', help='Output directory')
        grade_parser.add_argument('--samples', '-s', type=int, default=200, help='Frames to sample per video')
        grade_parser.add_argument('--max-iter', type=int, default=3, help='Maximum optimization iterations')
        grade_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
        grade_parser.add_argument('--dry-run', action='store_true', help='Show what would be processed')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze videos without grading')
        analyze_parser.add_argument('--input', '-i', required=True, help='Input directory with videos')
        analyze_parser.add_argument('--output', '-o', default='logs', help='Output directory for analysis')

        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return

        # Execute command
        if parsed_args.command == 'grade':
            self._run_grade(parsed_args)
        elif parsed_args.command == 'analyze':
            self._run_analyze(parsed_args)

    def _run_grade(self, args):
        """Run the grading pipeline."""
        print("🎨 ColorFlow - Dynamic Color Grading Pipeline")
        print("=" * 50)

        # Setup directories
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find videos
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(input_dir.glob(f'**/*{ext}')))

        if not video_files:
            print(f"❌ No video files found in {input_dir}")
            return

        print(f"📹 Found {len(video_files)} video files")
        for video in video_files:
            print(f"  - {video.name}")

        if args.dry_run:
            print("🔍 Dry run complete")
            return

        # Initialize components
        sampler = VideoSampler(self.config)
        analyzer = ColorAnalyzer(self.config)
        calibrator = ColorCalibrator(self.config)
        processor = VideoProcessor(self.config)
        visualizer = ColorVisualizer(output_dir=self.config.get('logging.file', 'logs').replace('.log', '/plots'))

        # Load LUT if provided
        lut_loader = LUTLoader()
        lut_data = None
        if args.lut and os.path.exists(args.lut):
            lut_data = lut_loader.load_cube(args.lut)
            if lut_data:
                print(f"🎯 Loaded LUT: {args.lut}")
            else:
                print(f"❌ Failed to load LUT: {args.lut}")

        # Process each video
        results = {
            'clips': {},
            'overall': {
                'total_videos': len(video_files),
                'converged_videos': 0,
                'total_time': 0,
                'mean_delta_e': 0
            }
        }

        total_start_time = time.time()

        for video_path in video_files:
            print(f"\n🎬 Processing: {video_path.name}")
            start_time = time.time()

            try:
                # Step 1: Sample frames
                print("  📊 Sampling frames...")
                frame_data = sampler.sample_frames(str(video_path))
                frames = [fd['frame'] for fd in frame_data]
                timestamps = [fd['timestamp'] for fd in frame_data]

                print(f"    Sampled {len(frames)} frames")

                # Step 2: Analyze frames
                print("  🔍 Analyzing color metrics...")
                frame_metrics = []
                for i, frame in enumerate(frames):
                    metrics = analyzer.analyze_frame(
                        frame, timestamps[i], i, str(video_path)
                    )
                    frame_metrics.append(metrics)

                # Step 3: Optimize transformation
                print("  ⚙️  Optimizing transformation...")
                optimization_result = calibrator.optimize_clip(
                    frames, timestamps, str(video_path)
                )

                print(f"    Final ΔE std: {optimization_result.delta_e_std:.".4f")
                print(f"    Iterations: {optimization_result.iteration}")
                print(f"    Converged: {'✅' if optimization_result.status.name == 'CONVERGED' else '❌'}")

                # Step 4: Apply transformation to full video
                print("  🎥 Applying transformation to video...")
                output_video_path = output_dir / f"graded_{video_path.name}"

                success = processor.process_video_with_params(
                    str(video_path),
                    str(output_video_path),
                    optimization_result.best_params
                )

                if success:
                    print(f"    ✅ Saved: {output_video_path}")
                else:
                    print(f"    ❌ Failed to process video")

                # Step 5: Generate visualizations
                print("  📈 Generating diagnostics...")
                visualizer.plot_frame_analysis(frame_metrics, f"Analysis - {video_path.name}")
                visualizer.plot_convergence_analysis(
                    analyzer.analyze_clip(frame_metrics),
                    f"Convergence - {video_path.name}"
                )

                # Store results
                processing_time = time.time() - start_time
                results['clips'][video_path.name] = {
                    'frame_count': len(frames),
                    'delta_e_std': optimization_result.delta_e_std,
                    'converged': optimization_result.status.name == 'CONVERGED',
                    'iterations': optimization_result.iteration,
                    'processing_time': processing_time,
                    'output_path': str(output_video_path) if success else None
                }

                if optimization_result.status.name == 'CONVERGED':
                    results['overall']['converged_videos'] += 1

            except Exception as e:
                print(f"  ❌ Error processing {video_path.name}: {e}")
                results['clips'][video_path.name] = {
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }

        # Overall statistics
        total_time = time.time() - total_start_time
        results['overall']['total_time'] = total_time

        if results['clips']:
            delta_e_values = [r['delta_e_std'] for r in results['clips'].values()
                            if 'delta_e_std' in r]
            if delta_e_values:
                results['overall']['mean_delta_e'] = sum(delta_e_values) / len(delta_e_values)

        # Generate summary report
        print("
📋 Generating summary report..."        summary_path = visualizer.create_summary_report(results)

        # Final results
        print("
🎉 Grading complete!"        print("=" * 50)
        print(f"Total videos processed: {len(video_files)}")
        print(f"Videos converged: {results['overall']['converged_videos']}")
        print(f"Total time: {total_time:.".2f"")
        print(f"Mean ΔE std: {results['overall'].get('mean_delta_e', 0):.4".4f"
        print(f"Summary report: {summary_path}")
        print(f"Output directory: {output_dir}")

        if results['overall']['converged_videos'] == len(video_files):
            print("✅ All videos converged successfully!")
        else:
            print(f"⚠️  {len(video_files) - results['overall']['converged_videos']} videos did not converge")

    def _run_analyze(self, args):
        """Run analysis only."""
        print("🔍 ColorFlow Analysis Mode")
        print("=" * 30)

        input_dir = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find videos
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(input_dir.glob(f'**/*{ext}')))

        if not video_files:
            print(f"❌ No video files found in {input_dir}")
            return

        # Initialize components
        sampler = VideoSampler(self.config)
        analyzer = ColorAnalyzer(self.config)
        visualizer = ColorVisualizer(output_dir=str(output_dir))

        for video_path in video_files:
            print(f"\n📹 Analyzing: {video_path.name}")

            # Sample frames
            frame_data = sampler.sample_frames(str(video_path))
            frames = [fd['frame'] for fd in frame_data]

            # Analyze frames
            frame_metrics = []
            for i, frame in enumerate(frames):
                metrics = analyzer.analyze_frame(
                    frame, frame_data[i]['timestamp'], i, str(video_path)
                )
                frame_metrics.append(metrics)

            # Generate analysis plots
            visualizer.plot_frame_analysis(frame_metrics, f"Analysis - {video_path.name}")

            # Clip-level analysis
            clip_metrics = analyzer.analyze_clip(frame_metrics)
            visualizer.plot_convergence_analysis(clip_metrics, f"Metrics - {video_path.name}")

            print(f"  Frames: {len(frames)}")
            print(f"  ΔE std: {clip_metrics.delta_e_std_global:.".4f")
            print(f"  Luminance range: {clip_metrics.luminance_range[0]:.3f} - {clip_metrics.luminance_range[1]:.3f".3f"
        print("
🔍 Analysis complete!"        print(f"Results saved to: {output_dir}")


def main():
    """Main entry point."""
    cli = ColorFlowCLI()
    cli.run()


if __name__ == "__main__":
    main()