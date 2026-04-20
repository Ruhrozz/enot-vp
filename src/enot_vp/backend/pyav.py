from __future__ import annotations
import warnings
from fractions import Fraction
from typing import Optional
from collections.abc import Iterator
import av
import cv2
import numpy as np


av.logging.set_level(av.logging.PANIC)


class PyAVInputBackend:
    def __init__(self, input_video, **kwargs) -> None:
        del kwargs
        self._container = av.open(input_video)
        self.stream = self._container.streams.video[0]
        self._frame_iterator = self._container.decode(self.stream)
        self._frame_count = self._get_frame_count(input_video)
        self.timestamp: float | None = None

    def _get_frame_count(self, input_video) -> int:
        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            warnings.warn("OpenCV failed to open the input video. Falling back to PyAV frame count.")
            return max(int(self.stream.frames), 0)

        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        if frame_count < 0:
            warnings.warn("OpenCV returned an invalid frame count. Falling back to PyAV frame count.")
            return max(int(self.stream.frames), 0)

        return frame_count

    def close(self):
        self._container.close()

    def __len__(self):
        return self._frame_count

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        try:
            video_frame = next(self._frame_iterator)
            if video_frame.pts is None:
                self.timestamp = None
            else:
                self.timestamp = float(video_frame.pts * video_frame.time_base)
            return video_frame.to_rgb().to_ndarray()
        except StopIteration:
            self._container.close()
            raise


class PyAVOutputBackend:
    REQUIRED_ARGS = ["rate", "width", "height"]

    def __init__(
        self,
        output_video,
        rate: Fraction,
        width: int,
        height: int,
        output_codec: str = "h264",
        pix_fmt: str = "yuv420p",
        **kwargs,
    ) -> None:
        del kwargs
        self._container = av.open(output_video, mode="w")
        self.stream = self._container.add_stream(
            codec_name=output_codec,
            rate=rate,
            width=width,
            height=height,
            options={"x265-params": "log_level=none"},
        )
        if not isinstance(self.stream, av.VideoStream):
            warnings.warn(f"Got {type(self.stream)} instead of {av.VideoStream}! It might not work as expected.")
        if isinstance(self.stream, av.VideoStream):
            self.stream.pix_fmt = pix_fmt

    @classmethod
    def from_input_backend(cls, output_video, input_backend: PyAVInputBackend, *args, **kwargs) -> PyAVOutputBackend:
        average_rate: Optional[Fraction] = input_backend.stream.average_rate

        if average_rate is None:
            warnings.warn("Video stream average rate is None! Using rate from kwargs.")
            rate = kwargs["rate"]
        else:
            rate = kwargs.get("rate", None) or average_rate

        return cls(
            output_video,
            rate=rate,
            width=kwargs.pop("width", None) or input_backend.stream.width,
            height=kwargs.pop("height", None) or input_backend.stream.height,
            *args,
            **kwargs,
        )

    def put(self, frame: np.ndarray):
        """Add video frame to output video."""
        if self._container is None:
            raise RuntimeError("Cannot create output video.")

        video_frame = av.VideoFrame.from_ndarray(frame)
        if isinstance(self.stream, av.VideoStream):
            for packet in self.stream.encode(video_frame):
                self._container.mux(packet)

    def close(self):
        # Flush any remaining frames in the encoder
        if isinstance(self.stream, av.VideoStream):
            for packet in self.stream.encode():
                self._container.mux(packet)
        self._container.close()
