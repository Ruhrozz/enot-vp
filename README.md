# ENOT Video Processor

Package for processing video with `pyav` package.
Provides methods for reading from video and writing modified video frames.

# Examples

Reading

```
from enot_vp import VideoProcessor
from ultralytics import YOLO

model = YOLO(MODEL_PATH)

with VideoProcessor(input_video=VIDEO_PATH) as vp:
    for frame in vp:
        ndarray = model(frame.to_image(), imgsz=1600, verbose=False)[0].plot()[..., ::-1]
```

Modification

```
from enot_vp import VideoProcessor
from ultralytics import YOLO

model = YOLO(MODEL_PATH)

with VideoProcessor(input_video=VIDEO_PATH, output_video=OUTPUT_VIDEO_PATH) as vp:
    for frame in vp:
        ndarray = model(frame.to_image(), imgsz=1600, verbose=False)[0].plot()[..., ::-1]
        vp.put(ndarray)
```
