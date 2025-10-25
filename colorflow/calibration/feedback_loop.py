"""
Closed-loop optimization system for color grading convergence.
Implements iterative refinement based on ΔE metrics and temporal consistency.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import get_config
from ..analysis.metrics import ColorAnalyzer, FrameMetrics
from ..transforms.dynamic_graph import DynamicColorTransform, TransformParams


class ConvergenceStatus(Enum):
    """Convergence status for optimization."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    CONVERGED = "converged"
    FAILED = "failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class OptimizationResult:
    """Result of optimization iteration."""
    iteration: int
    status: ConvergenceStatus
    delta_e_std: float
    improvement: float  # Change in ΔE std from previous iteration
    best_params: List[TransformParams]
    metrics: Dict[str, float]


class ColorCalibrator:
    """Closed-loop color calibration system."""

    def __init__(self, config=None):
        """Initialize calibrator with configuration."""
        self.config = config or get_config()
        self.analyzer = ColorAnalyzer(config)
        self.transform = DynamicColorTransform(config)

        # Optimization parameters
        self.max_iterations = self.config.max_iterations
        self.delta_e_threshold = self.config.delta_e_threshold
        self.ev_ema_alpha = self.config.ev_ema_alpha
        self.wb_ema_alpha = self.config.wb_ema_alpha
        self.saturation_ema_alpha = self.config.saturation_ema_alpha

        # State tracking
        self.iteration_history = []
        self.best_delta_e = float('inf')
        self.best_params = None

    def optimize_clip(self, frames: List[np.ndarray],
                     timestamps: List[float] = None,
                     video_path: str = "") -> OptimizationResult:
        """Run complete optimization loop for a video clip."""

        if not frames:
            return OptimizationResult(
                iteration=0,
                status=ConvergenceStatus.FAILED,
                delta_e_std=float('inf'),
                improvement=0.0,
                best_params=[],
                metrics={}
            )

        if timestamps is None:
            timestamps = list(range(len(frames)))

        # Iteration 1: Baseline analysis
        frame_metrics = []
        for i, frame in enumerate(frames):
            metrics = self.analyzer.analyze_frame(
                frame, timestamps[i], i, video_path
            )
            frame_metrics.append(metrics)

        baseline_delta_e = self._calculate_clip_delta_e_std(frame_metrics)

        # Initialize optimization
        current_params = self._initialize_parameters(frame_metrics)

        for iteration in range(1, self.max_iterations + 1):
            print(f"Iteration {iteration}/{self.max_iterations}")

            # Apply current transformation
            transformed_frames = []
            for i, frame in enumerate(frames):
                transformed = self.transform.apply_transform(frame, current_params[i])
                transformed_frames.append(transformed)

            # Analyze transformed frames
            new_metrics = []
            for i, frame in enumerate(transformed_frames):
                metrics = self.analyzer.analyze_frame(
                    frame, timestamps[i], i, video_path
                )
                new_metrics.append(metrics)

            # Calculate improvement
            new_delta_e = self._calculate_clip_delta_e_std(new_metrics)
            improvement = baseline_delta_e - new_delta_e

            print(f"  ΔE std: {new_delta_e".4f"}, Improvement: {improvement".4f"}")

            # Check convergence
            if new_delta_e < self.delta_e_threshold:
                return OptimizationResult(
                    iteration=iteration,
                    status=ConvergenceStatus.CONVERGED,
                    delta_e_std=new_delta_e,
                    improvement=improvement,
                    best_params=current_params,
                    metrics=self._extract_metrics(new_metrics)
                )

            # Update parameters for next iteration
            current_params = self._refine_parameters(current_params, new_metrics, frame_metrics)

            # Store iteration result
            self.iteration_history.append(OptimizationResult(
                iteration=iteration,
                status=ConvergenceStatus.IN_PROGRESS,
                delta_e_std=new_delta_e,
                improvement=improvement,
                best_params=current_params.copy(),
                metrics=self._extract_metrics(new_metrics)
            ))

        # Max iterations reached
        return OptimizationResult(
            iteration=self.max_iterations,
            status=ConvergenceStatus.MAX_ITERATIONS,
            delta_e_std=new_delta_e,
            improvement=improvement,
            best_params=current_params,
            metrics=self._extract_metrics(new_metrics)
        )

    def _initialize_parameters(self, frame_metrics: List[FrameMetrics]) -> List[TransformParams]:
        """Initialize transformation parameters based on frame analysis."""
        params = []

        for i, metrics in enumerate(frame_metrics):
            # Simple heuristic initialization
            ev_adjustment = self._calculate_ev_from_metrics(metrics)
            wb_matrix = self._calculate_wb_from_metrics(metrics)
            saturation_gain = self._calculate_saturation_from_metrics(metrics)

            params.append(TransformParams(
                frame_index=i,
                timestamp=metrics.timestamp,
                ev_adjustment=ev_adjustment,
                wb_matrix=wb_matrix,
                tone_shadows=0.0,
                tone_midtones=0.0,
                tone_highlights=0.0,
                saturation_gain=saturation_gain,
                lut_strength=0.0
            ))

        return params

    def _calculate_ev_from_metrics(self, metrics: FrameMetrics) -> float:
        """Calculate initial EV adjustment from frame metrics."""
        # Aim for middle gray at 50% luminance
        current_luma = metrics.luminance_percentiles[2]  # P50
        target_luma = 0.5

        if current_luma > 0:
            ev = np.log2(target_luma / current_luma)
            return np.clip(ev, -2.0, 2.0)  # Conservative initial range

        return 0.0

    def _calculate_wb_from_metrics(self, metrics: FrameMetrics) -> np.ndarray:
        """Calculate initial white balance matrix from metrics."""
        # Simple WB based on gray world assumption
        mean_rgb = metrics.mean_rgb

        # Normalize to green channel
        if mean_rgb[1] > 0:
            wb_gains = mean_rgb[1] / mean_rgb
            wb_gains = np.clip(wb_gains, 0.5, 2.0)  # Limit WB correction
        else:
            wb_gains = np.array([1.0, 1.0, 1.0])

        # Create diagonal WB matrix
        return np.diag(wb_gains)

    def _calculate_saturation_from_metrics(self, metrics: FrameMetrics) -> float:
        """Calculate initial saturation adjustment."""
        # Target moderate saturation
        current_sat = metrics.saturation_mean
        target_sat = 0.3  # Moderate saturation

        if current_sat > 0:
            gain = target_sat / current_sat
            return np.clip(gain, 0.7, 1.5)  # Conservative saturation adjustment

        return 1.0

    def _refine_parameters(self, current_params: List[TransformParams],
                          new_metrics: List[FrameMetrics],
                          original_metrics: List[FrameMetrics]) -> List[TransformParams]:
        """Refine parameters based on new analysis."""
        refined_params = []

        for i, (params, new_metric, orig_metric) in enumerate(zip(
            current_params, new_metrics, original_metrics
        )):
            # Adjust EV based on luminance error
            luma_error = new_metric.luminance_percentiles[2] - 0.5
            ev_adjust = params.ev_adjustment - 0.5 * luma_error  # Proportional feedback

            # Adjust WB based on color temperature
            rgb_balance = new_metric.mean_rgb / np.sum(new_metric.mean_rgb)
            target_balance = np.array([0.3, 0.4, 0.3])  # Target RGB balance

            wb_error = rgb_balance - target_balance
            wb_matrix = params.wb_matrix.copy()

            # Small adjustments to WB matrix
            for j in range(3):
                wb_matrix[j, j] *= (1 + 0.1 * wb_error[j])

            # Adjust saturation based on ΔE
            delta_e_error = new_metric.delta_e_mean - 1.0  # Target ΔE around 1.0
            sat_gain = params.saturation_gain * (1 - 0.1 * delta_e_error)

            refined_params.append(TransformParams(
                frame_index=params.frame_index,
                timestamp=params.timestamp,
                ev_adjustment=np.clip(ev_adjust, -3.0, 3.0),
                wb_matrix=wb_matrix,
                tone_shadows=params.tone_shadows,
                tone_midtones=params.tone_midtones,
                tone_highlights=params.tone_highlights,
                saturation_gain=np.clip(sat_gain, 0.5, 2.0),
                lut_strength=params.lut_strength
            ))

        # Apply temporal smoothing
        return [self.transform.smooth_parameters(p) for p in refined_params]

    def _calculate_clip_delta_e_std(self, frame_metrics: List[FrameMetrics]) -> float:
        """Calculate standard deviation of ΔE across all frames."""
        delta_e_values = [fm.delta_e_mean for fm in frame_metrics
                         if fm.delta_e_mean is not None]

        if not delta_e_values:
            return float('inf')

        return np.std(delta_e_values)

    def _extract_metrics(self, frame_metrics: List[FrameMetrics]) -> Dict[str, float]:
        """Extract summary metrics from frame metrics."""
        if not frame_metrics:
            return {}

        delta_e_values = [fm.delta_e_mean for fm in frame_metrics
                         if fm.delta_e_mean is not None]

        ev_values = [fm.ev_estimate for fm in frame_metrics
                    if fm.ev_estimate is not None]

        saturation_values = [fm.saturation_mean for fm in frame_metrics]

        return {
            'mean_delta_e': np.mean(delta_e_values) if delta_e_values else 0,
            'std_delta_e': np.std(delta_e_values) if delta_e_values else 0,
            'mean_ev': np.mean(ev_values) if ev_values else 0,
            'ev_range': (min(ev_values), max(ev_values)) if ev_values else (0, 0),
            'mean_saturation': np.mean(saturation_values),
            'saturation_range': (min(saturation_values), max(saturation_values))
        }