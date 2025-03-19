# WiLoR_Keypoints:

Module to make just the keypoint calculation. Stripped a little bit from original.
You need to Downlaod some weight files and put it into a folder.
You must set the weights folder in initialization:

```python
from wilor_keypoints import WILOR

model = WILOR(weights_folder="./wilor_weights")
# load an image
result = model.predict(image)
```

## Install

`pip install git+https://github.com/Schmetzler/wilor_keypoints.git`

### Needed models (put everything in a folder)
* YOLO Detector (detector.pt) (can be found [here](https://huggingface.co/warmshao/WiLoR-mini/blob/main/pretrained_models/detector.pt) \[I renamed it to `yolo_hands.pt`\])
* `wilor.safetensors` file (you may also use the original ckpt file)
* `MANO_RIGHT.pkl` can be found [here](https://mano.is.tue.mpg.de/) (I will not supply it as you should register on the website to download the model)
* and `mano_mean_params.npz`

I put everything together in a 7z file (wilor_model_weights.7z) on my Google Drive (besides MANO_RIGHT.pkl) so you can download everything from [there](https://drive.google.com/drive/folders/1hfLQhse5DP460Q-j0d-vG_obCVIsc9Bt?usp=drive_link).

I use `torch.float8_e4m3fn` format for `wilor.safetensors` to save space, after loading it is transformed back to float16 again.

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
