from typing import Optional, Union
import av
import numpy as np


class VideoProcessor:
    def __init__(self, input_video: str, output_video: Optional[str] = None, output_codec: str = "hevc"):
        self.input_video = input_video
        self.output_video = output_video
        self._output_codec = output_codec
        self.input_container = None
        self.output_container = None
        self.input_stream = None
        self.output_stream = None
        self._frame_iterator = None

    def __enter__(self):
        # Open input video
        self.input_container = av.open(self.input_video)
        self.input_stream = self.input_container.streams.video[0]

        # Set up output if specified
        if self.output_video:
            self.output_container = av.open(self.output_video, mode="w")
            self.output_stream = self.output_container.add_stream(
                codec_name=self._output_codec,
                rate=self.input_stream.average_rate,
                width=self.input_stream.width,
                height=self.input_stream.height,
            )
            self.output_stream.pix_fmt = "yuv420p"

        # Create frame iterator with progress bar
        self._frame_iterator = self.input_container.decode(self.input_stream)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_tb

        if self.output_container:
            # Flush any remaining frames in the encoder
            for packet in self.output_stream.encode():
                self.output_container.mux(packet)
            self.output_container.close()

        self.input_container.close()

        if exc_type is not None:
            print(f"Error occurred: {exc_val}")

        return True

    def __iter__(self):
        return self

    def __next__(self):
        if self._frame_iterator is None:
            raise RuntimeError("VideoProcessor not initialized. Use in a 'with' block.")

        try:
            return next(self._frame_iterator)
        except StopIteration:
            self._frame_iterator.close()
            raise

    def put(self, processed_frame: Union[av.VideoFrame, np.ndarray]):
        """Add video frame to output video."""
        if self.output_container is None:
            raise RuntimeError("No output video specified in constructor")

        # Convert frame to AV VideoFrame and encode
        if not isinstance(processed_frame, av.VideoFrame):
            processed_frame = av.VideoFrame.from_ndarray(processed_frame)

        for packet in self.output_stream.encode(processed_frame):
            self.output_container.mux(packet)
