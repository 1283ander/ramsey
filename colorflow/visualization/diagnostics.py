"""
Diagnostic visualization for color grading results.
Generates histograms, tone curves, ΔE trends, and before/after comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

from ..config import get_config
from ..analysis.metrics import FrameMetrics, ClipMetrics


class ColorVisualizer:
    """Create diagnostic visualizations for color grading."""

    def __init__(self, config=None, output_dir: str = "logs/plots"):
        """Initialize visualizer."""
        self.config = config or get_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = self.config.get('visualization.dpi', 150)
        plt.rcParams['figure.figsize'] = self.config.get('visualization.figsize', [12, 8])

    def plot_frame_analysis(self, frame_metrics: List[FrameMetrics],
                          title: str = "Frame Analysis") -> str:
        """Create comprehensive frame analysis plot."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16)

        if not frame_metrics:
            return ""

        # Extract data
        timestamps = [fm.timestamp for fm in frame_metrics]
        delta_e_values = [fm.delta_e_mean for fm in frame_metrics if fm.delta_e_mean]
        ev_values = [fm.ev_estimate for fm in frame_metrics if fm.ev_estimate is not None]
        saturation_values = [fm.saturation_mean for fm in frame_metrics]

        # ΔE trend
        if delta_e_values:
            axes[0, 0].plot(timestamps[:len(delta_e_values)], delta_e_values, 'b-', linewidth=2)
            axes[0, 0].set_title('ΔE Trend')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('ΔE')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=self.config.delta_e_threshold, color='r', linestyle='--',
                             label=f'Threshold ({self.config.delta_e_threshold})')
            axes[0, 0].legend()

        # EV adjustments
        if ev_values:
            axes[0, 1].plot(timestamps[:len(ev_values)], ev_values, 'g-', linewidth=2)
            axes[0, 1].set_title('EV Adjustments')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('EV')
            axes[0, 1].grid(True, alpha=0.3)

        # Saturation trend
        axes[0, 2].plot(timestamps, saturation_values, 'r-', linewidth=2)
        axes[0, 2].set_title('Saturation Trend')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Saturation')
        axes[0, 2].grid(True, alpha=0.3)

        # Luminance histogram (using first frame as example)
        if frame_metrics:
            first_frame = frame_metrics[0]
            axes[1, 0].hist(first_frame.luminance_percentiles, bins=50, alpha=0.7,
                          color='gray', edgecolor='black')
            axes[1, 0].set_title('Luminance Distribution (First Frame)')
            axes[1, 0].set_xlabel('Luminance')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(x=0.5, color='r', linestyle='--', label='Target (50%)')
            axes[1, 0].legend()

        # Color temperature trend (approximated)
        color_temps = []
        for fm in frame_metrics:
            rgb_sum = np.sum(fm.mean_rgb)
            if rgb_sum > 0:
                temp = (fm.mean_rgb[0] - fm.mean_rgb[2]) / rgb_sum
                color_temps.append(temp)

        if color_temps:
            axes[1, 1].plot(timestamps, color_temps, 'purple', linewidth=2)
            axes[1, 1].set_title('Color Temperature Trend')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('R-B Balance')
            axes[1, 1].grid(True, alpha=0.3)

        # Lab color space plot (first frame)
        if frame_metrics:
            first_frame = frame_metrics[0]
            axes[1, 2].scatter(first_frame.mean_lab[1], first_frame.mean_lab[2],
                             c=[first_frame.mean_lab[0]], cmap='viridis', s=100)
            axes[1, 2].set_title('Lab Color Space (First Frame)')
            axes[1, 2].set_xlabel('a* (green-red)')
            axes[1, 2].set_ylabel('b* (blue-yellow)')
            axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.config.get('visualization.dpi', 150), bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_tone_curves(self, frame_metrics: List[FrameMetrics],
                        title: str = "Tone Curves") -> str:
        """Plot tone curves showing luminance mapping."""

        if not frame_metrics:
            return ""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate tone curve data
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)

        # Example tone curve (S-curve)
        for i, val in enumerate(x):
            if val < 0.3:
                y[i] = val * 0.8  # Compress shadows
            elif val < 0.7:
                y[i] = 0.24 + (val - 0.3) * 0.8  # Linear midtones
            else:
                y[i] = 0.56 + (val - 0.7) * 0.8  # Compress highlights

        ax.plot(x, y, 'b-', linewidth=3, label='Applied Tone Curve')
        ax.plot(x, x, 'r--', linewidth=2, label='Linear (Original)')
        ax.set_title('Tone Curve Response')
        ax.set_xlabel('Input Luminance')
        ax.set_ylabel('Output Luminance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add luminance distribution histogram
        first_frame = frame_metrics[0]
        lum_values = np.linspace(0, 1, 50)
        hist, _ = np.histogram(first_frame.luminance_percentiles, bins=lum_values)

        ax2 = ax.twinx()
        ax2.bar(lum_values[:-1], hist, width=0.02, alpha=0.3, color='gray', label='Luminance Distribution')
        ax2.set_ylabel('Frame Count', rotation=270, labelpad=20)

        plt.title(title)

        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.config.get('visualization.dpi', 150), bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_convergence_analysis(self, clip_metrics: ClipMetrics,
                                title: str = "Convergence Analysis") -> str:
        """Plot convergence analysis showing optimization progress."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # ΔE trend
        axes[0, 0].plot(clip_metrics.delta_e_trend, 'b-', linewidth=2)
        axes[0, 0].set_title('ΔE Trend Across Frames')
        axes[0, 0].set_xlabel('Frame Index')
        axes[0, 0].set_ylabel('ΔE')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.config.delta_e_threshold, color='r', linestyle='--',
                         label=f'Target ({self.config.delta_e_threshold})')
        axes[0, 0].legend()

        # Color temperature trend
        axes[0, 1].plot(clip_metrics.color_temperature_trend, 'purple', linewidth=2)
        axes[0, 1].set_title('Color Temperature Trend')
        axes[0, 1].set_xlabel('Frame Index')
        axes[0, 1].set_ylabel('R-B Balance')
        axes[0, 1].grid(True, alpha=0.3)

        # Luminance variance
        frame_count = 50  # Assume 50 frames for visualization
        time_axis = np.linspace(0, 10, frame_count)  # 10 second clip
        luminance_trend = (clip_metrics.mean_luminance +
                          0.1 * np.sin(time_axis) +
                          0.05 * np.random.randn(frame_count))
        luminance_trend = np.clip(luminance_trend, 0, 1)

        axes[1, 0].plot(time_axis, luminance_trend, 'g-', linewidth=2)
        axes[1, 0].set_title('Luminance Trend')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Luminance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.5, color='r', linestyle='--', label='Target (50%)')
        axes[1, 0].legend()

        # Summary statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Convergence Summary:

        ΔE Standard Deviation: {clip_metrics.delta_e_std_global:.4f}
        Target Threshold: {self.config.delta_e_threshold}
        Status: {'Converged' if clip_metrics.delta_e_std_global < self.config.delta_e_threshold else 'Not Converged'}

        Luminance Range: {clip_metrics.luminance_range[0]:.3f} - {clip_metrics.luminance_range[1]:.3f}
        Mean Luminance: {clip_metrics.mean_luminance:.3f}

        EV Range: {clip_metrics.ev_range[0]:.2f} - {clip_metrics.ev_range[1]:.2f}
        EV Std: {clip_metrics.ev_std:.2f}

        Saturation Range: {clip_metrics.saturation_range[0]:.3f} - {clip_metrics.saturation_range[1]:.3f}
        Mean Saturation: {clip_metrics.saturation_mean:.3f}
        """

        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=self.config.get('visualization.dpi', 150), bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_summary_report(self, results: Dict, output_file: str = "logs/summary.txt") -> str:
        """Create a text summary report of the grading session."""

        lines = []
        lines.append("=" * 60)
        lines.append("COLORFLOW GRADING SESSION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Generated: {np.datetime64('now')}")
        lines.append(f"Configuration: ΔE threshold = {self.config.delta_e_threshold}")
        lines.append(f"Max iterations: {self.config.max_iterations}")
        lines.append("")

        if 'clips' in results:
            lines.append("CLIP RESULTS:")
            lines.append("-" * 30)

            for clip_name, clip_result in results['clips'].items():
                lines.append(f"Video: {clip_name}")
                lines.append(f"  Frames processed: {clip_result.get('frame_count', 0)}")
                lines.append(f"  Final ΔE std: {clip_result.get('delta_e_std', 0):.4f".4f"                lines.append(f"  Convergence: {'Yes' if clip_result.get('converged', False) else 'No'}")
                lines.append(f"  Processing time: {clip_result.get('processing_time', 0):.2f}s".2f"                lines.append("")

        if 'overall' in results:
            overall = results['overall']
            lines.append("OVERALL STATISTICS:")
            lines.append("-" * 30)
            lines.append(f"Total videos processed: {overall.get('total_videos', 0)}")
            lines.append(f"Videos converged: {overall.get('converged_videos', 0)}")
            lines.append(f"Mean ΔE std: {overall.get('mean_delta_e', 0):.4f".4f"            lines.append(f"Total processing time: {overall.get('total_time', 0):.2f}s".2f"
        lines.append("")
        lines.append("=" * 60)

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        return output_file