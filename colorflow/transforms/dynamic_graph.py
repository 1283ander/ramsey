"""
Dynamic color transformation graph for adaptive frame-by-frame grading.
Implements the parametric pipeline: RGB_in → Exposure → WB → ToneCurve → Saturation → LUT → RGB_out
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2

from ..config import get_config


@dataclass
class TransformParams:
    """Parameters for a single frame transformation."""
    frame_index: int
    timestamp: float

    # Exposure adjustment
    ev_adjustment: float

    # White balance
    wb_matrix: np.ndarray  # 3x3 matrix

    # Tone curve parameters (S-curve)
    tone_shadows: float    # Shadow adjustment (-1 to 1)
    tone_midtones: float   # Midtone adjustment (-1 to 1)
    tone_highlights: float # Highlight adjustment (-1 to 1)

    # Saturation
    saturation_gain: float

    # LUT blending
    lut_strength: float    # 0 to 1


class DynamicColorTransform:
    """Dynamic color transformation pipeline."""

    def __init__(self, config=None):
        """Initialize transformation pipeline."""
        self.config = config or get_config()

        # Load transform limits
        self.ev_min = self.config.get('transform_limits.ev_min', -3.0)
        self.ev_max = self.config.get('transform_limits.ev_max', 3.0)
        self.wb_shift_max = self.config.get('transform_limits.wb_shift_max', 0.1)
        self.saturation_min = self.config.get('transform_limits.saturation_min', 0.5)
        self.saturation_max = self.config.get('transform_limits.saturation_max', 2.0)

        # Temporal smoothing factors
        self.ev_ema_alpha = self.config.ev_ema_alpha
        self.wb_ema_alpha = self.config.wb_ema_alpha
        self.saturation_ema_alpha = self.config.saturation_ema_alpha

        # Previous frame parameters for temporal smoothing
        self._prev_params = None

    def apply_transform(self, frame: np.ndarray, params: TransformParams) -> np.ndarray:
        """Apply complete transformation pipeline to a frame."""
        # Ensure frame is in correct format
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be RGB with shape (H, W, 3)")

        # Convert to float32 for processing
        rgb = frame.astype(np.float32) / 255.0

        # Apply transformation pipeline
        rgb = self._apply_exposure(rgb, params.ev_adjustment)
        rgb = self._apply_white_balance(rgb, params.wb_matrix)
        rgb = self._apply_tone_curve(rgb, params)
        rgb = self._apply_saturation(rgb, params.saturation_gain)

        # Clamp and convert back
        rgb = np.clip(rgb, 0, 1)
        return (rgb * 255).astype(np.uint8)

    def _apply_exposure(self, rgb: np.ndarray, ev: float) -> np.ndarray:
        """Apply exposure adjustment."""
        ev = np.clip(ev, self.ev_min, self.ev_max)
        multiplier = 2.0 ** ev
        return rgb * multiplier

    def _apply_white_balance(self, rgb: np.ndarray, wb_matrix: np.ndarray) -> np.ndarray:
        """Apply white balance transformation matrix."""
        # Apply 3x3 matrix transformation
        rgb_linear = np.dot(rgb.reshape(-1, 3), wb_matrix.T)
        return rgb_linear.reshape(rgb.shape)

    def _apply_tone_curve(self, rgb: np.ndarray, params: TransformParams) -> np.ndarray:
        """Apply S-curve tone mapping."""
        # Convert to luminance
        luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

        # Apply S-curve transformation
        shadows = params.tone_shadows
        midtones = params.tone_midtones
        highlights = params.tone_highlights

        # Piecewise tone curve
        result = np.zeros_like(luminance)

        # Shadows (0-0.3)
        shadow_mask = luminance < 0.3
        result[shadow_mask] = luminance[shadow_mask] * (1 + shadows * 0.5)

        # Midtones (0.3-0.7)
        midtone_mask = (luminance >= 0.3) & (luminance < 0.7)
        result[midtone_mask] = luminance[midtone_mask] + midtones * 0.2 * (luminance[midtone_mask] - 0.5)

        # Highlights (0.7-1.0)
        highlight_mask = luminance >= 0.7
        result[highlight_mask] = luminance[highlight_mask] + highlights * 0.3 * (1 - luminance[highlight_mask])

        # Apply to each channel proportionally
        scale = result / (luminance + 1e-8)  # Avoid division by zero
        scale = np.clip(scale, 0.5, 2.0)  # Limit scaling

        return rgb * scale[:, :, np.newaxis]

    def _apply_saturation(self, rgb: np.ndarray, saturation_gain: float) -> np.ndarray:
        """Apply saturation adjustment."""
        saturation_gain = np.clip(saturation_gain, self.saturation_min, self.saturation_max)

        # Convert to HSL, modify saturation, convert back
        hsl = self._rgb_to_hsl_batch(rgb)
        hsl[:, :, 1] *= saturation_gain  # Modify saturation channel
        hsl[:, :, 1] = np.clip(hsl[:, :, 1], 0, 1)

        return self._hsl_to_rgb_batch(hsl)

    def smooth_parameters(self, new_params: TransformParams) -> TransformParams:
        """Apply temporal smoothing to parameters using EMA."""
        if self._prev_params is None:
            self._prev_params = new_params
            return new_params

        # Apply EMA smoothing
        smoothed = TransformParams(
            frame_index=new_params.frame_index,
            timestamp=new_params.timestamp,
            ev_adjustment=self._ema_smooth(self._prev_params.ev_adjustment,
                                         new_params.ev_adjustment,
                                         self.ev_ema_alpha),
            wb_matrix=self._ema_smooth_matrix(self._prev_params.wb_matrix,
                                            new_params.wb_matrix,
                                            self.wb_ema_alpha),
            tone_shadows=self._ema_smooth(self._prev_params.tone_shadows,
                                        new_params.tone_shadows,
                                        0.3),
            tone_midtones=self._ema_smooth(self._prev_params.tone_midtones,
                                         new_params.tone_midtones,
                                         0.3),
            tone_highlights=self._ema_smooth(self._prev_params.tone_highlights,
                                           new_params.tone_highlights,
                                           0.3),
            saturation_gain=self._ema_smooth(self._prev_params.saturation_gain,
                                           new_params.saturation_gain,
                                           self.saturation_ema_alpha),
            lut_strength=self._ema_smooth(self._prev_params.lut_strength,
                                        new_params.lut_strength,
                                        0.3)
        )

        self._prev_params = smoothed
        return smoothed

    def _ema_smooth(self, prev: float, current: float, alpha: float) -> float:
        """Apply exponential moving average smoothing."""
        return alpha * current + (1 - alpha) * prev

    def _ema_smooth_matrix(self, prev: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
        """Apply EMA smoothing to transformation matrix."""
        return alpha * current + (1 - alpha) * prev

    def generate_identity_params(self, frame_index: int, timestamp: float) -> TransformParams:
        """Generate identity transformation parameters."""
        return TransformParams(
            frame_index=frame_index,
            timestamp=timestamp,
            ev_adjustment=0.0,
            wb_matrix=np.eye(3),
            tone_shadows=0.0,
            tone_midtones=0.0,
            tone_highlights=0.0,
            saturation_gain=1.0,
            lut_strength=0.0
        )

    def _rgb_to_hsl_batch(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSL (vectorized version)."""
        rgb = np.clip(rgb, 0, 1)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val

        # Lightness
        l = (max_val + min_val) / 2

        # Saturation
        s = np.where(delta == 0, 0, delta / (2 - max_val - min_val))

        # Hue
        h = np.zeros_like(delta)

        # Red is max
        red_mask = (max_val == r) & (delta != 0)
        h[red_mask] = (60 * ((g[red_mask] - b[red_mask]) / delta[red_mask]) + 360) % 360

        # Green is max
        green_mask = (max_val == g) & (delta != 0)
        h[green_mask] = (60 * ((b[green_mask] - r[green_mask]) / delta[green_mask]) + 120) % 360

        # Blue is max
        blue_mask = (max_val == b) & (delta != 0)
        h[blue_mask] = (60 * ((r[blue_mask] - g[blue_mask]) / delta[blue_mask]) + 240) % 360

        return np.stack([h, s, l], axis=2)

    def _hsl_to_rgb_batch(self, hsl: np.ndarray) -> np.ndarray:
        """Convert HSL to RGB (vectorized version)."""
        h, s, l = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = l - c / 2

        h60 = h / 60

        # Initialize RGB channels
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        # Hue sectors
        sector0 = (h60 >= 0) & (h60 < 1)
        r[sector0] = c[sector0]
        g[sector0] = x[sector0]
        b[sector0] = 0

        sector1 = (h60 >= 1) & (h60 < 2)
        r[sector1] = x[sector1]
        g[sector1] = c[sector1]
        b[sector1] = 0

        sector2 = (h60 >= 2) & (h60 < 3)
        r[sector2] = 0
        g[sector2] = c[sector2]
        b[sector2] = x[sector2]

        sector3 = (h60 >= 3) & (h60 < 4)
        r[sector3] = 0
        g[sector3] = x[sector3]
        b[sector3] = c[sector3]

        sector4 = (h60 >= 4) & (h60 < 5)
        r[sector4] = x[sector4]
        g[sector4] = 0
        b[sector4] = c[sector4]

        sector5 = (h60 >= 5) & (h60 < 6)
        r[sector5] = c[sector5]
        g[sector5] = 0
        b[sector5] = x[sector5]

        return np.stack([r + m, g + m, b + m], axis=2)