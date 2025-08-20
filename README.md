# ENOT Video Processor

Package for processing video with `pyav` package.
Provides methods for reading from video and writing modified video frames.

# Examples

## Reading Only

```python
from enot_vp import VideoProcessor
import numpy as np

with VideoProcessor(input_video=VIDEO_PATH) as vp:
    for frame in vp:
        print(np.mean(frame))
```

## Reading and Writing

```python
# TODO: does it work?
from enot_vp import VideoProcessor
from ultralytics import YOLO

model = YOLO(MODEL_PATH)

with VideoProcessor(input_video=VIDEO_PATH, output_video=OUTPUT_VIDEO_PATH) as vp:
    for frame in vp:
        ndarray = model(frame, imgsz=1600, verbose=False)[0].plot()[..., ::-1]
        vp.put(ndarray)
```

## Writing Only

```python
from enot_vp import VideoProcessor
import cv2, random, numpy as np

# `width`, `height` and `rate` are required arguments here
with VideoProcessor(output_video=OUTPUT_VIDEO_PATH, width=128, height=128, rate=30) as vp:
    for i in range(200):
        vp.put(cv2.circle(
            np.zeros((128,128,3), dtype=np.uint8), 
            (random.randint(20,108), random.randint(20,108)),
            random.randint(5,20),
            (255,255,255),
            -1,
        ))
```
