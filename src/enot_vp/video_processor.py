import warnings
import numpy as np
from enot_vp.backend.pyav import PyAVInputBackend, PyAVOutputBackend


class VideoProcessor:
    def __init__(
        self,
        *args,
        input_backend: type[PyAVInputBackend] = PyAVInputBackend,
        output_backend: type[PyAVOutputBackend] = PyAVOutputBackend,
        **kwargs,
    ):
        if len(args) != 0:
            raise RuntimeError("Class takes no positional arguments! Use keyword arguments.")

        self.input_backend = None
        self.output_backend = None

        if "input_video" in kwargs:
            self.input_backend = input_backend(**kwargs)

        mask = [arg in kwargs for arg in output_backend.REQUIRED_ARGS]
        unspecified_args = [arg for specified, arg in zip(mask, output_backend.REQUIRED_ARGS) if not specified]

        if "output_video" in kwargs:
            if all(mask):
                self.output_backend = output_backend(**kwargs)
            elif self.input_backend is not None:
                warnings.warn(
                    f"{unspecified_args} were not specified in kwargs. "
                    f"Using the same parameters as in `input_video`"
                )
                self.output_backend = output_backend.from_input_backend(input_backend=self.input_backend, **kwargs)
            else:
                raise RuntimeError(f"Found required but unspecified arguments: {unspecified_args}")

    def put(self, frame: np.ndarray):
        if self.output_backend is None:
            raise NotImplementedError("Method `put` is implemented only for `output_video`")
        self.output_backend.put(frame)

    def __len__(self):
        return len(self.input_backend) if self.input_backend is not None else 0

    def __enter__(self):
        return self

    def __iter__(self):
        if self.input_backend is None:
            raise NotImplementedError("Iteration is implemented only for `input_video`")
        return iter(self.input_backend)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.output_backend is not None:
            self.output_backend.close()
        if self.input_backend is not None:
            self.input_backend.close()
        return False
