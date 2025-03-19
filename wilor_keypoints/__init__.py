from skimage.filters import gaussian
import torch
import cv2
from ultralytics import YOLO
import os
import numpy as np
from tqdm import tqdm
import logging

from .utils import WeightsOnlyFalse
from .utils.logger import get_logger
from .models.wilor import WiLor
from .utils import utils
from safetensors.torch import load_model

class WILOR:
    def __init__(self, weights_folder, device=torch.device("cpu"), dtype=torch.float16, verbose=True, init_models=True):
        self.verbose = verbose
        if self.verbose:
            self.logger = get_logger(self.__class__.__name__, lv=logging.INFO)
        else:
            self.logger = get_logger(self.__class__.__name__, lv=logging.ERROR)
        
        self.FOCAL_LENGTH = 5000
        self.IMAGE_SIZE = 256
        
        self.wilor_weight_path = os.path.join(weights_folder, "wilor.safetensors")
        self.mano_mean_path = os.path.join(weights_folder, "mano_mean_params.npz")
        self.mano_model_path = os.path.join(weights_folder, "MANO_RIGHT.pkl")
        self.yolo_weight_path = os.path.join(weights_folder, "yolo_hands.pt")

        if init_models:
            self.init_models(device, dtype)

    def init_models(self, device, dtype):
        self.device = device
        self.dtype = dtype

        self.wilor_model = WiLor(
            mano_model_path=self.mano_model_path, mano_mean_path=self.mano_mean_path,
            focal_length=self.FOCAL_LENGTH, image_size=self.IMAGE_SIZE
        )

        if self.wilor_weight_path.endswith(".safetensors"):
            if device is None:
                device_ = "cpu"
            elif isinstance(device, torch.device):
                if not device.index is None:
                    device_ = device.index
                else:
                    device_ = device.type
            else:
                device_ = device
            load_model(self.wilor_model, self.wilor_weight_path, strict=False, device=device_)
        else:
            try:
                state = torch.load(self.wilor_weight_path, weights_only=True)
                self.wilor_model.load_state_dict(state, strict=False)
            except Exception:
                state = torch.load(self.wilor_weight_path, weights_only=False)["state_dict"]
                self.wilor_model.load_state_dict(state, strict=False)
        self.wilor_model.eval()
        self.wilor_model.to(dtype=dtype)

        with WeightsOnlyFalse():
            self.hand_detector = YOLO(self.yolo_weight_path)
        self.hand_detector.to(device)
        

    @torch.no_grad()
    def predict(self, image, **kwargs):
        self.logger.info("start hand detection >>> ")
        detections = self.hand_detector(image, conf=kwargs.get("hand_conf", 0.3), verbose=self.verbose)[0]
        detect_rets = []
        bboxes = []
        is_rights = []
        for det in detections:
            hand_bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_rights.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(hand_bbox[:4].tolist())
            detect_rets.append({"hand_bbox": bboxes[-1], "is_right": is_rights[-1]})

        if len(bboxes) == 0:
            self.logger.warn("No hand detected!")
            return detect_rets

        bboxes = np.stack(bboxes)

        rescale_factor = kwargs.get("rescale_factor", 2.5)
        center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        self.logger.info(f"detect {bboxes.shape[0]} hands")
        self.logger.info("start hand 3d pose estimation >>> ")
        img_patches = []
        img_size = np.array([image.shape[1], image.shape[0]])
        for i in tqdm(range(bboxes.shape[0]), disable=not self.verbose):
            bbox_size = scale[i].max()
            patch_width = patch_height = self.IMAGE_SIZE
            right = is_rights[i]
            flip = right == 0
            box_center = center[i]

            cvimg = image.copy()
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size * 1.0) / patch_width)
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

            img_patch_cv, trans = utils.generate_image_patch_cv2(cvimg,
                                                                 box_center[0], box_center[1],
                                                                 bbox_size, bbox_size,
                                                                 patch_width, patch_height,
                                                                 flip, 1.0, 0,
                                                                 border_mode=cv2.BORDER_CONSTANT)
            img_patches.append(img_patch_cv)
        img_patches = np.stack(img_patches)
        img_patches = torch.from_numpy(img_patches).to(device=self.device, dtype=self.dtype)
        wilor_output = self.wilor_model(img_patches)
        wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

        for i in range(len(detect_rets)):
            wilor_output_i = {key: val[[i]] for key, val in wilor_output.items()}
            pred_cam = wilor_output_i["pred_cam"]
            bbox_size = scale[i].max()
            box_center = center[i]
            right = is_rights[i]
            multiplier = (2 * right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
                wilor_output_i["pred_vertices"][:, :, 0] = -wilor_output_i["pred_vertices"][:, :, 0]
                wilor_output_i["global_orient"] = np.concatenate(
                    (wilor_output_i["global_orient"][:, :, 0:1], -wilor_output_i["global_orient"][:, :, 1:3]),
                    axis=-1)
                wilor_output_i["hand_pose"] = np.concatenate(
                    (wilor_output_i["hand_pose"][:, :, 0:1], -wilor_output_i["hand_pose"][:, :, 1:3]),
                    axis=-1)
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(pred_cam, box_center[None], bbox_size, img_size[None],
                                                     scaled_focal_length)
            wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
            wilor_output_i["scaled_focal_length"] = scaled_focal_length
            pred_keypoints_2d = utils.perspective_projection(wilor_output_i["pred_keypoints_3d"],
                                                             translation=pred_cam_t_full,
                                                             focal_length=np.array([scaled_focal_length] * 2)[None],
                                                             camera_center=img_size[None] / 2)
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i

        self.logger.info("finish detection!")
        return detect_rets

    @torch.no_grad()
    def predict_with_bboxes(self, image, bboxes, is_rights, **kwargs):
        self.logger.info("Predict with hand bboxes Input >>> ")
        detect_rets = []
        if len(bboxes) == 0:
            self.logger.warn("No hand detected!")
            return detect_rets
        for i in range(bboxes.shape[0]):
            detect_rets.append({"hand_bbox": bboxes[i, :4].tolist(), "is_right": is_rights[i]})
        rescale_factor = kwargs.get("rescale_factor", 2.5)
        center = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2.0
        scale = rescale_factor * (bboxes[:, 2:4] - bboxes[:, 0:2])
        self.logger.info(f"detect {bboxes.shape[0]} hands")
        self.logger.info("start hand 3d pose estimation >>> ")
        img_patches = []
        img_size = np.array([image.shape[1], image.shape[0]])
        for i in tqdm(range(bboxes.shape[0]), disable=not self.verbose):
            bbox_size = scale[i].max()
            patch_width = patch_height = self.IMAGE_SIZE
            right = is_rights[i]
            flip = right == 0
            box_center = center[i]

            cvimg = image.copy()
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size * 1.0) / patch_width)
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)
            img_size = np.array([cvimg.shape[1], cvimg.shape[0]])

            img_patch_cv, trans = utils.generate_image_patch_cv2(cvimg,
                                                                 box_center[0], box_center[1],
                                                                 bbox_size, bbox_size,
                                                                 patch_width, patch_height,
                                                                 flip, 1.0, 0,
                                                                 border_mode=cv2.BORDER_CONSTANT)
            img_patches.append(img_patch_cv)

        img_patches = np.stack(img_patches)
        img_patches = torch.from_numpy(img_patches).to(device=self.device, dtype=self.dtype)
        wilor_output = self.wilor_model(img_patches)
        wilor_output = {k: v.cpu().float().numpy() for k, v in wilor_output.items()}

        for i in range(len(detect_rets)):
            wilor_output_i = {key: val[[i]] for key, val in wilor_output.items()}
            pred_cam = wilor_output_i["pred_cam"]
            bbox_size = scale[i].max()
            box_center = center[i]
            right = is_rights[i]
            multiplier = (2 * right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            if right == 0:
                wilor_output_i["pred_keypoints_3d"][:, :, 0] = -wilor_output_i["pred_keypoints_3d"][:, :, 0]
                wilor_output_i["pred_vertices"][:, :, 0] = -wilor_output_i["pred_vertices"][:, :, 0]
                wilor_output_i["global_orient"] = np.concatenate(
                    (wilor_output_i["global_orient"][:, :, 0:1], -wilor_output_i["global_orient"][:, :, 1:3]),
                    axis=-1)
                wilor_output_i["hand_pose"] = np.concatenate(
                    (wilor_output_i["hand_pose"][:, :, 0:1], -wilor_output_i["hand_pose"][:, :, 1:3]),
                    axis=-1)
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(pred_cam, box_center[None], bbox_size, img_size[None],
                                                     scaled_focal_length)
            wilor_output_i["pred_cam_t_full"] = pred_cam_t_full
            wilor_output_i["scaled_focal_length"] = scaled_focal_length
            pred_keypoints_2d = utils.perspective_projection(wilor_output_i["pred_keypoints_3d"],
                                                             translation=pred_cam_t_full,
                                                             focal_length=np.array([scaled_focal_length] * 2)[None],
                                                             camera_center=img_size[None] / 2)
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i

        self.logger.info("finish detection!")
        return detect_rets