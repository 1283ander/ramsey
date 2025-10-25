"""
Efficient video I/O tools using ffmpeg-python and PyAV for frame sampling and processing.
Handles video metadata, audio preservation, and high-quality output encoding.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Iterator
import ffmpeg
import av
from PIL import Image
from pathlib import Path
import json

from ..config import get_config


class VideoSampler:
    """Efficient video frame sampler using ffmpeg and PyAV."""

    def __init__(self, config=None):
        """Initialize video sampler with configuration."""
        self.config = config or get_config()
        self.samples_per_video = self.config.sampling_frames_per_video

    def get_video_info(self, video_path: str) -> Dict:
        """Get comprehensive video metadata."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')

            return {
                'duration': float(probe['format']['duration']),
                'frame_count': int(video_stream.get('nb_frames', 0)),
                'fps': float(video_stream.get('avg_frame_rate', '30/1').split('/')[0]) /
                       float(video_stream.get('avg_frame_rate', '30/1').split('/')[1]),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'codec': video_stream['codec_name'],
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'has_audio': any(s['codec_type'] == 'audio' for s in probe['streams']),
                'format': probe['format']['format_name']
            }
        except Exception as e:
            print(f"Warning: Could not probe video {video_path}: {e}")
            return {}

    def sample_frames_evenly(self, video_path: str, n_frames: int = None) -> List[np.ndarray]:
        """Sample frames evenly distributed throughout the video."""
        if n_frames is None:
            n_frames = self.samples_per_video

        info = self.get_video_info(video_path)
        if not info:
            return []

        duration = info['duration']
        if duration <= 0:
            return []

        # Use PyAV for efficient frame extraction
        frames = []
        timestamps = np.linspace(0, duration, n_frames)

        container = av.open(video_path)
        video_stream = container.streams.video[0]

        for timestamp in timestamps:
            # Seek to timestamp
            container.seek(int(timestamp * av.time_base))

            # Get frame at this timestamp
            for frame in container.decode(video=0):
                if frame.time >= timestamp:
                    # Convert to numpy array
                    img = frame.to_image()
                    frame_array = np.array(img)

                    # Convert RGB to BGR for OpenCV compatibility if needed
                    if len(frame_array.shape) == 3:
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                    frames.append(frame_array)
                    break

        container.close()
        return frames

    def sample_frames_random(self, video_path: str, n_frames: int = None) -> List[np.ndarray]:
        """Sample frames randomly throughout the video."""
        if n_frames is None:
            n_frames = self.samples_per_video

        info = self.get_video_info(video_path)
        if not info:
            return []

        duration = info['duration']
        if duration <= 0:
            return []

        # Generate random timestamps
        np.random.seed(42)  # For reproducibility
        timestamps = np.random.uniform(0, duration, n_frames)

        frames = []
        container = av.open(video_path)
        video_stream = container.streams.video[0]

        for timestamp in sorted(timestamps):
            container.seek(int(timestamp * av.time_base))

            for frame in container.decode(video=0):
                if frame.time >= timestamp:
                    img = frame.to_image()
                    frame_array = np.array(img)
                    if len(frame_array.shape) == 3:
                        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    frames.append(frame_array)
                    break

        container.close()
        return frames

    def sample_frames(self, video_path: str, method: str = "even") -> List[Dict]:
        """Sample frames with timestamps and metadata."""
        if method == "even":
            frames = self.sample_frames_evenly(video_path)
        elif method == "random":
            frames = self.sample_frames_random(video_path)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        info = self.get_video_info(video_path)
        duration = info.get('duration', 0)

        # Add metadata to each frame
        frame_data = []
        for i, frame in enumerate(frames):
            timestamp = (i / len(frames)) * duration if duration > 0 else i
            frame_data.append({
                'frame': frame,
                'timestamp': timestamp,
                'frame_index': i,
                'video_path': video_path
            })

        return frame_data

    def process_video(self, input_path: str, output_path: str,
                     transform_func=None, audio_passthrough: bool = True) -> bool:
        """Process video with frame-by-frame transformation."""
        try:
            info = self.get_video_info(input_path)
            if not info:
                return False

            # Build ffmpeg command
            stream = ffmpeg.input(input_path)

            if transform_func:
                # For now, use a simple approach - we'll implement proper frame processing later
                # This is a placeholder for the actual video processing pipeline
                pass

            # Output with preserved audio and metadata
            output_args = {
                'c:v': 'libx265',
                'c:a': 'copy' if audio_passthrough else 'aac',
                's': '1920x1080',  # 1080p as requested
                'crf': 23,
                'preset': 'medium',
                'pix_fmt': 'yuv420p10le'
            }

            if audio_passthrough and info.get('has_audio'):
                output_args['c:a'] = 'copy'

            output = ffmpeg.output(stream, output_path, **output_args)
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            return True

        except Exception as e:
            print(f"Error processing video {input_path}: {e}")
            return False


class LUTLoader:
    """Load and parse .cube LUT files."""

    def __init__(self):
        self.luts = {}

    def load_cube(self, lut_path: str) -> Optional[Dict]:
        """Load a .cube LUT file and return interpolation data."""
        try:
            with open(lut_path, 'r') as f:
                lines = f.readlines()

            # Parse header
            header = {}
            data_start = 0

            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith('TITLE'):
                    header['title'] = line.split('"')[1] if '"' in line else line.split()[-1]
                elif line.startswith('LUT_3D_SIZE'):
                    header['size'] = int(line.split()[-1])
                elif line.startswith('DOMAIN_MIN'):
                    header['domain_min'] = float(line.split()[-1])
                elif line.startswith('DOMAIN_MAX'):
                    header['domain_max'] = float(line.split()[-1])
                elif line.startswith('LUT_1D_SIZE'):
                    header['lut_1d_size'] = int(line.split()[-1])
                else:
                    # Data starts here
                    data_start = i
                    break

            if 'size' not in header:
                raise ValueError("LUT_3D_SIZE not found in .cube file")

            # Parse 3D LUT data
            size = header['size']
            expected_lines = size * size * size

            lut_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                values = [float(x) for x in line.split()]
                if len(values) >= 3:
                    lut_data.append(values[:3])  # R, G, B

            if len(lut_data) != expected_lines:
                raise ValueError(f"Expected {expected_lines} LUT entries, got {len(lut_data)}")

            # Reshape to 3D array
            lut_3d = np.array(lut_data).reshape((size, size, size, 3))

            # Set defaults
            header.setdefault('domain_min', 0.0)
            header.setdefault('domain_max', 1.0)

            return {
                'header': header,
                'lut_3d': lut_3d,
                'size': size,
                'domain_min': header['domain_min'],
                'domain_max': header['domain_max']
            }

        except Exception as e:
            print(f"Error loading LUT {lut_path}: {e}")
            return None

    def interpolate_3d(self, lut_data: Dict, rgb_input: np.ndarray) -> np.ndarray:
        """Interpolate 3D LUT for given RGB values."""
        size = lut_data['size']
        domain_min = lut_data['domain_min']
        domain_max = lut_data['domain_max']
        lut_3d = lut_data['lut_3d']

        # Normalize input to LUT domain
        rgb_norm = np.clip(rgb_input, 0, 1)
        rgb_norm = domain_min + rgb_norm * (domain_max - domain_min)

        # Scale to LUT indices
        scale = (size - 1) / (domain_max - domain_min)
        indices = rgb_norm * scale
        indices = np.clip(indices, 0, size - 2)

        # Get integer and fractional parts
        indices_floor = np.floor(indices).astype(int)
        indices_frac = indices - indices_floor

        # Trilinear interpolation
        result = np.zeros_like(rgb_input)

        for i in range(3):  # R, G, B channels
            for c in range(3):  # Output R, G, B
                # Get 8 corner values
                corners = np.zeros((2, 2, 2))

                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            r_idx = np.clip(indices_floor[:, 0] + di, 0, size - 1)
                            g_idx = np.clip(indices_floor[:, 1] + dj, 0, size - 1)
                            b_idx = np.clip(indices_floor[:, 2] + dk, 0, size - 1)

                            corners[di, dj, dk] = lut_3d[r_idx, g_idx, b_idx, c]

                # Trilinear interpolation
                c00 = corners[0, 0, 0] * (1 - indices_frac[:, 0]) + corners[1, 0, 0] * indices_frac[:, 0]
                c01 = corners[0, 0, 1] * (1 - indices_frac[:, 0]) + corners[1, 0, 1] * indices_frac[:, 0]
                c10 = corners[0, 1, 0] * (1 - indices_frac[:, 0]) + corners[1, 1, 0] * indices_frac[:, 0]
                c11 = corners[0, 1, 1] * (1 - indices_frac[:, 0]) + corners[1, 1, 1] * indices_frac[:, 0]

                c0 = c00 * (1 - indices_frac[:, 1]) + c10 * indices_frac[:, 1]
                c1 = c01 * (1 - indices_frac[:, 1]) + c11 * indices_frac[:, 1]

                result[:, c] = c0 * (1 - indices_frac[:, 2]) + c1 * indices_frac[:, 2]

        return np.clip(result, 0, 1)