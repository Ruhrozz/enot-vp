from __future__ import annotations
import warnings
from fractions import Fraction
from typing import Optional
from collections.abc import Iterator
import av
import numpy as np


class PyAVInputBackend:
    def __init__(self, input_video, **kwargs) -> None:
        del kwargs
        self._container = av.open(input_video)
        self.stream = self._container.streams.video[0]
        self._frame_iterator = self._container.decode(self.stream)

    def close(self):
        self._container.close()
    
    def __len__(self):
        return self._container.streams.video[0].frames

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        try:
            video_frame = next(self._frame_iterator)
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
