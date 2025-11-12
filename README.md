# WiLoR_Keypoints:

Module to make just the keypoint calculation. Stripped a little bit from original.
You need to Downlaod some weight files and put it into a folder.
You must set the weights folder in initialization:

```python
from wilor_keypoints import WILOR

model = WILOR(weights_folder="./weights_folder")
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

The file structure should look like this:
```
|--weights_folder
   |--wilor.safetensors
   |--MANO_RIGHT.pkl
   |--mano_mean_params.npz
   |--yolo_hands.pt
```
You can change the models if you do something like this:

```python
import torch
from wilor_keypoints import WILOR
model = WILOR("weights_folder/", init_models=False)
model.wilor_weight_path = <PATH_TO_OTHER_FILE>
model.yolo_weight_path = <PATH_TO_OTHER_YOLO_CHECKPOINT_FILE>
# you have to init the models manually
model.init_models(device="cpu", dtype=torch.float16)
```

## Funding

The work was conducted as part of the research project **KI-StudiUm** at the [**Westsächsische Hochschule Zwickau**](https://www.whz.de/english/), which was funded by the [**Federal Ministry of Research, Technology and Space**](https://www.bmftr.bund.de/EN/Home/home_node.html) as part of the federal-state initiative "KI in der Hochschulbildung" under the funding code `16DHBKI063`.

<picture>
     <source srcset="assets/bmftr-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/bmftr-en-light.svg" height="75px">
</picture>
<picture>
     <source srcset="assets/whz-en-dark.svg" media="(prefers-color-scheme: dark)">
     <img src="assets/whz-en-light.svg" height="75px">
</picture>
