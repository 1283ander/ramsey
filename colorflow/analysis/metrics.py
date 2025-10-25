"""
Color metrics computation for frame analysis and ΔE calculations.
Supports multiple color spaces: RGB, HSL, Lab, JzAzBz for comprehensive analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import ndimage
import cv2

from ..config import get_config


@dataclass
class FrameMetrics:
    """Color metrics for a single frame."""
    timestamp: float
    frame_index: int
    video_path: str

    # Basic statistics
    mean_rgb: np.ndarray  # [R, G, B] mean values
    std_rgb: np.ndarray   # [R, G, B] standard deviations

    # Luminance metrics (P1, P5, P50, P95, P99 percentiles)
    luminance_percentiles: np.ndarray

    # HSL statistics
    mean_hsl: np.ndarray   # [H, S, L] mean values
    saturation_mean: float
    saturation_std: float

    # Lab color space metrics
    mean_lab: np.ndarray   # [L, a, b] mean values
    std_lab: np.ndarray    # [L, a, b] standard deviations

    # JzAzBz HDR perceptual space (if computed)
    mean_jzazbz: Optional[np.ndarray] = None
    std_jzazbz: Optional[np.ndarray] = None

    # Delta-E metrics (compared to target)
    delta_e_mean: Optional[float] = None
    delta_e_std: Optional[float] = None
    delta_e_max: Optional[float] = None

    # Clipping information
    clipped_pixels: int = 0
    total_pixels: int = 0

    # Exposure value estimate
    ev_estimate: Optional[float] = None


@dataclass
class ClipMetrics:
    """Aggregated metrics for an entire video clip."""
    video_path: str
    frame_count: int

    # Global statistics
    mean_luminance: float
    luminance_range: Tuple[float, float]

    # Color consistency metrics
    delta_e_std_global: float  # Standard deviation of ΔE across all frames
    delta_e_trend: np.ndarray  # ΔE trend over time

    # Temporal stability
    luminance_variance: float
    color_temperature_trend: np.ndarray

    # Exposure consistency
    ev_range: Tuple[float, float]
    ev_std: float

    # Saturation analysis
    saturation_mean: float
    saturation_range: Tuple[float, float]


class ColorAnalyzer:
    """Comprehensive color analysis for video frames."""

    def __init__(self, config=None):
        """Initialize color analyzer with configuration."""
        self.config = config or get_config()
        self.metrics_space = self.config.metrics_space

        # Target neutral colors for ΔE calculation
        self.target_lab = np.array([
            self.config.get('color_targets.neutral_lab.L', 50.0),
            self.config.get('color_targets.neutral_lab.a', 0.0),
            self.config.get('color_targets.neutral_lab.b', 0.0)
        ])

        self.target_jzazbz = np.array([
            self.config.get('color_targets.neutral_jzazbz.Jz', 0.5),
            self.config.get('color_targets.neutral_jzazbz.Az', 0.0),
            self.config.get('color_targets.neutral_jzazbz.Bz', 0.0)
        ])

    def analyze_frame(self, frame: np.ndarray, timestamp: float = 0.0,
                     frame_index: int = 0, video_path: str = "") -> FrameMetrics:
        """Analyze a single frame and compute comprehensive color metrics."""

        # Ensure frame is in correct format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be RGB image with shape (H, W, 3)")

        height, width = frame.shape[:2]
        total_pixels = height * width

        # Convert to float32 for calculations
        frame_float = frame.astype(np.float32) / 255.0

        # Basic RGB statistics
        mean_rgb = np.mean(frame_float, axis=(0, 1))
        std_rgb = np.std(frame_float, axis=(0, 1))

        # Luminance calculation (using green channel as proxy)
        luminance = frame_float[:, :, 1]  # Green channel for luminance
        luminance_percentiles = np.percentile(luminance, [1, 5, 50, 95, 99])

        # HSL conversion and statistics
        hsl = self._rgb_to_hsl_batch(frame_float)
        mean_hsl = np.mean(hsl, axis=(0, 1))
        saturation_mean = np.mean(hsl[:, :, 1])
        saturation_std = np.std(hsl[:, :, 1])

        # Lab color space conversion
        lab = self._rgb_to_lab_batch(frame_float)
        mean_lab = np.mean(lab, axis=(0, 1))
        std_lab = np.std(lab, axis=(0, 1))

        # JzAzBz conversion (HDR perceptual space)
        jzazbz = self._rgb_to_jzazbz_batch(frame_float)
        mean_jzazbz = np.mean(jzazbz, axis=(0, 1))
        std_jzazbz = np.std(jzazbz, axis=(0, 1))

        # ΔE calculations
        if self.metrics_space == "lab":
            delta_e_values = self._calculate_delta_e_lab(lab.reshape(-1, 3))
            target = self.target_lab
        else:  # jzazbz
            delta_e_values = self._calculate_delta_e_jzazbz(jzazbz.reshape(-1, 3))
            target = self.target_jzazbz

        delta_e_mean = np.mean(delta_e_values)
        delta_e_std = np.std(delta_e_values)
        delta_e_max = np.max(delta_e_values)

        # Clipping detection
        clipped_pixels = np.sum(
            (frame_float < 0.01) | (frame_float > 0.99)
        )

        # EV estimation (rough approximation)
        ev_estimate = self._estimate_ev(luminance)

        return FrameMetrics(
            timestamp=timestamp,
            frame_index=frame_index,
            video_path=video_path,
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            luminance_percentiles=luminance_percentiles,
            mean_hsl=mean_hsl,
            saturation_mean=saturation_mean,
            saturation_std=saturation_std,
            mean_lab=mean_lab,
            std_lab=std_lab,
            mean_jzazbz=mean_jzazbz,
            std_jzazbz=std_jzazbz,
            delta_e_mean=delta_e_mean,
            delta_e_std=delta_e_std,
            delta_e_max=delta_e_max,
            clipped_pixels=clipped_pixels,
            total_pixels=total_pixels,
            ev_estimate=ev_estimate
        )

    def analyze_clip(self, frame_metrics: List[FrameMetrics]) -> ClipMetrics:
        """Analyze a collection of frame metrics for clip-level insights."""

        if not frame_metrics:
            raise ValueError("No frame metrics provided")

        # Extract luminance values
        luminances = [fm.luminance_percentiles[2] for fm in frame_metrics]  # P50
        mean_luminance = np.mean(luminances)
        luminance_range = (min(luminances), max(luminances))

        # Global ΔE statistics
        delta_e_values = [fm.delta_e_mean for fm in frame_metrics]
        delta_e_std_global = np.std(delta_e_values)

        # ΔE trend over time
        delta_e_trend = np.array(delta_e_values)

        # Temporal stability metrics
        luminance_variance = np.var(luminances)

        # Color temperature trend (approximated from RGB ratios)
        color_temps = []
        for fm in frame_metrics:
            # Simple color temperature estimation from RGB ratios
            rgb_sum = np.sum(fm.mean_rgb)
            if rgb_sum > 0:
                temp = (fm.mean_rgb[0] - fm.mean_rgb[2]) / rgb_sum
                color_temps.append(temp)
        color_temperature_trend = np.array(color_temps)

        # EV analysis
        ev_values = [fm.ev_estimate for fm in frame_metrics if fm.ev_estimate is not None]
        ev_range = (min(ev_values), max(ev_values)) if ev_values else (0, 0)
        ev_std = np.std(ev_values) if ev_values else 0

        # Saturation analysis
        saturations = [fm.saturation_mean for fm in frame_metrics]
        saturation_mean = np.mean(saturations)
        saturation_range = (min(saturations), max(saturations))

        return ClipMetrics(
            video_path=frame_metrics[0].video_path,
            frame_count=len(frame_metrics),
            mean_luminance=mean_luminance,
            luminance_range=luminance_range,
            delta_e_std_global=delta_e_std_global,
            delta_e_trend=delta_e_trend,
            luminance_variance=luminance_variance,
            color_temperature_trend=color_temperature_trend,
            ev_range=ev_range,
            ev_std=ev_std,
            saturation_mean=saturation_mean,
            saturation_range=saturation_range
        )

    def _rgb_to_hsl_batch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSL color space."""
        # Normalize to [0, 1]
        rgb = np.clip(rgb, 0, 1)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val

        # Lightness
        l = (max_val + min_val) / 2

        # Saturation
        s = np.zeros_like(delta)
        mask = delta != 0
        s[mask] = delta[mask] / (2 - max_val[mask] - min_val[mask]) if np.max(l) <= 0.5 else delta[mask] / (2 - delta[mask])

        # Hue
        h = np.zeros_like(delta)
        mask_r = (max_val == r) & (delta != 0)
        mask_g = (max_val == g) & (delta != 0)
        mask_b = (max_val == b) & (delta != 0)

        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

        return np.stack([h, s, l], axis=2)

    def _rgb_to_lab_batch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to Lab color space."""
        # Apply gamma correction
        rgb_linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

        # RGB to XYZ transformation (sRGB to XYZ D65)
        transform = np.array([
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505]
        ])

        xyz = np.dot(rgb_linear, transform.T)

        # Normalize for D65 illuminant
        xyz_ref = np.array([0.95047, 1.0, 1.08883])
        xyz_normalized = xyz / xyz_ref

        # XYZ to Lab conversion
        def f(t):
            return np.where(t > 0.008856, t ** (1/3), (903.3 * t + 16) / 116)

        fx = f(xyz_normalized[:, :, 0])
        fy = f(xyz_normalized[:, :, 1])
        fz = f(xyz_normalized[:, :, 2])

        l = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)

        return np.stack([l, a, b], axis=2)

    def _rgb_to_jzazbz_batch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to JzAzBz HDR perceptual color space."""
        # This is a simplified version - full implementation would be more complex
        # For now, using Lab as approximation
        return self._rgb_to_lab_batch(rgb) * 0.1  # Scale down for JzAzBz range

    def _calculate_delta_e_lab(self, lab_values: np.ndarray) -> np.ndarray:
        """Calculate ΔE (CIEDE2000) between Lab values and target."""
        # Simplified ΔE calculation (Euclidean distance in Lab space)
        # Full CIEDE2000 would be more accurate but more complex
        target = self.target_lab.reshape(1, -1)
        return np.sqrt(np.sum((lab_values - target) ** 2, axis=1))

    def _calculate_delta_e_jzazbz(self, jzazbz_values: np.ndarray) -> np.ndarray:
        """Calculate ΔE in JzAzBz space."""
        target = self.target_jzazbz.reshape(1, -1)
        return np.sqrt(np.sum((jzazbz_values - target) ** 2, axis=1))

    def _estimate_ev(self, luminance: np.ndarray) -> float:
        """Estimate exposure value from luminance."""
        # Rough EV estimation based on middle gray (18% gray = EV 0)
        middle_gray = 0.18
        mean_lum = np.mean(luminance)

        if mean_lum > 0:
            ev = np.log2(mean_lum / middle_gray)
            return np.clip(ev, -10, 10)  # Clamp to reasonable range
        return 0.0