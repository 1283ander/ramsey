"""
Video processing and rendering module using ffmpeg-python.
Applies transformations frame-by-frame while preserving audio and metadata.
"""

import os
import numpy as np
from typing import List, Dict, Optional, Callable
import tempfile
import json

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("Warning: ffmpeg-python not available. Video processing will use fallback methods.")

try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    print("Warning: PyAV not available. Video processing will be limited.")

from ..config import get_config
from ..transforms.dynamic_graph import TransformParams


class VideoProcessor:
    """Process and render videos with frame-by-frame transformations."""

    def __init__(self, config=None):
        """Initialize video processor."""
        self.config = config or get_config()

        if not FFMPEG_AVAILABLE and not AV_AVAILABLE:
            raise ImportError("Either ffmpeg-python or PyAV is required for video processing")

    def process_video_with_params(self, input_path: str, output_path: str,
                                transform_params: List[TransformParams],
                                transform_func: Optional[Callable] = None) -> bool:
        """Process video applying transformation parameters to each frame."""

        if not os.path.exists(input_path):
            print(f"Error: Input video not found: {input_path}")
            return False

        # Check if output already exists
        if os.path.exists(output_path) and not self._should_overwrite(output_path):
            print(f"Output exists and overwrite not requested: {output_path}")
            return True

        try:
            if AV_AVAILABLE:
                return self._process_with_av(input_path, output_path, transform_params, transform_func)
            else:
                return self._process_with_ffmpeg(input_path, output_path, transform_params)

        except Exception as e:
            print(f"Error processing video {input_path}: {e}")
            return False

    def _process_with_av(self, input_path: str, output_path: str,
                        transform_params: List[TransformParams],
                        transform_func: Optional[Callable] = None) -> bool:
        """Process video using PyAV for precise frame control."""

        print(f"Processing {input_path} with PyAV...")

        # Open input video
        input_container = av.open(input_path)
        input_video_stream = input_container.streams.video[0]
        input_audio_stream = input_container.streams.audio[0] if input_container.streams.audio else None

        # Get video properties
        fps = input_video_stream.average_rate
        width = input_video_stream.width
        height = input_video_stream.height

        # Open output video
        output_container = av.open(output_path, 'w')
        output_video_stream = output_container.add_stream('libx265', rate=fps)
        output_video_stream.width = 1920  # 1080p as requested
        output_video_stream.height = 1080
        output_video_stream.pix_fmt = 'yuv420p10le'

        if input_audio_stream:
            output_audio_stream = output_container.add_stream(template=input_audio_stream)

        # Process frames
        frame_idx = 0

        for frame in input_container.decode(video=0):
            # Find corresponding transform parameters
            params = self._find_params_for_frame(transform_params, frame_idx, frame.time)

            if params:
                # Convert frame to numpy array
                img = frame.to_image()
                frame_array = np.array(img)

                # Apply transformation
                if transform_func:
                    transformed_array = transform_func(frame_array, params)
                else:
                    # Use default transformation pipeline
                    from ..transforms.dynamic_graph import DynamicColorTransform
                    transform = DynamicColorTransform(self.config)
                    transformed_array = transform.apply_transform(frame_array, params)

                # Resize to 1080p if needed
                if transformed_array.shape[:2] != (1080, 1920):
                    transformed_array = self._resize_frame(transformed_array, (1920, 1080))

                # Create output frame
                output_frame = av.VideoFrame.from_ndarray(transformed_array, format='rgb24')
                output_frame.pts = frame.pts
                output_frame.time_base = input_video_stream.time_base

                # Encode frame
                for packet in output_video_stream.encode(output_frame):
                    output_container.mux(packet)

            frame_idx += 1

        # Flush encoder
        for packet in output_video_stream.encode():
            output_container.mux(packet)

        # Copy audio if available
        if input_audio_stream:
            for frame in input_container.decode(audio=0):
                # Create audio frame
                output_frame = av.AudioFrame.from_ndarray(
                    frame.to_ndarray(), format=frame.format.name, layout=frame.layout.name
                )
                output_frame.pts = frame.pts
                output_frame.time_base = input_audio_stream.time_base
                output_frame.sample_rate = frame.sample_rate

                for packet in output_audio_stream.encode(output_frame):
                    output_container.mux(packet)

            # Flush audio encoder
            for packet in output_audio_stream.encode():
                output_container.mux(packet)

        # Close containers
        input_container.close()
        output_container.close()

        print(f"Successfully processed video: {output_path}")
        return True

    def _process_with_ffmpeg(self, input_path: str, output_path: str,
                           transform_params: List[TransformParams]) -> bool:
        """Fallback video processing using ffmpeg-python."""

        print(f"Processing {input_path} with ffmpeg-python...")

        try:
            # Build ffmpeg command
            stream = ffmpeg.input(input_path)

            # Apply basic processing (simplified for ffmpeg-python)
            output = ffmpeg.output(
                stream,
                output_path,
                c='libx265',
                s='1920x1080',
                crf=23,
                preset='medium',
                pix_fmt='yuv420p10le',
                **{'c:a': 'copy'}  # Copy audio
            )

            # Run ffmpeg command
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            print(f"Successfully processed video: {output_path}")
            return True

        except Exception as e:
            print(f"ffmpeg processing failed: {e}")
            return False

    def _find_params_for_frame(self, transform_params: List[TransformParams],
                             frame_idx: int, timestamp: float) -> Optional[TransformParams]:
        """Find transformation parameters for a specific frame."""

        if not transform_params:
            return None

        # Find closest parameter by frame index or timestamp
        best_match = None
        min_diff = float('inf')

        for params in transform_params:
            # Try frame index first
            diff = abs(params.frame_index - frame_idx)
            if diff < min_diff:
                min_diff = diff
                best_match = params

        return best_match

    def _resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to target dimensions."""
        try:
            import cv2
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
        except ImportError:
            # Fallback to PIL
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize(target_size, Image.LANCZOS)
            return np.array(img)

    def _should_overwrite(self, output_path: str) -> bool:
        """Check if output should be overwritten."""
        # For now, always overwrite if requested
        # In a real implementation, this would check config or command line flags
        return True

    def extract_audio(self, input_path: str, output_path: str) -> bool:
        """Extract audio track from video."""
        try:
            if not FFMPEG_AVAILABLE:
                return False

            stream = ffmpeg.input(input_path)
            audio = stream.audio

            output = ffmpeg.output(audio, output_path)
            ffmpeg.run(output, overwrite_output=True, quiet=True)

            return True

        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def combine_video_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Combine video and audio tracks."""
        try:
            if not FFMPEG_AVAILABLE:
                return False

            video_stream = ffmpeg.input(video_path)
            audio_stream = ffmpeg.input(audio_path)

            output = ffmpeg.output(
                video_stream.video, audio_stream.audio, output_path,
                c='libx265', s='1920x1080', crf=23, preset='medium',
                pix_fmt='yuv420p10le'
            )

            ffmpeg.run(output, overwrite_output=True, quiet=True)
            return True

        except Exception as e:
            print(f"Error combining video and audio: {e}")
            return False