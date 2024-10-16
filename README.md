## WiLoR-mini: Simplifying WiLoR into a Python package

**Original repository: [WiLoR](https://github.com/rolpotamias/WiLoR), thanks to the authors for sharing**

I have simplified WiLoR, focusing on the inference process. Now it can be installed via pip and used directly, and it will automatically download the model.

### How to use?
* install: `pip install git+https://github.com/warmshao/WiLoR-mini`
* Usage:
```python
import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
img_path = "assets/img.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = pipe.predict(image)

```
For more usage examples, please refer to: `tests/test_pipelines.py`

### Demo
<video src="https://github.com/user-attachments/assets/ca7329fe-0b66-4eb6-87a5-4cb5cbe9ec43" controls="controls" width="300" height="500">您的浏览器不支持播放该视频！</video>