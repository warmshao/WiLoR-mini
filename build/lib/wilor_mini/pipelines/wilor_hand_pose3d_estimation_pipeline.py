# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: wilor_hand_pose3d_estimation_pipeline.py
import pdb
from skimage.filters import gaussian
import torch
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import numpy as np
from tqdm import tqdm
import logging

from ..utils.logger import get_logger
from ..models.wilor import WiLor
from ..utils import utils


class WiLorHandPose3dEstimationPipeline:
    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", True)
        if self.verbose:
            self.logger = get_logger(self.__class__.__name__, lv=logging.INFO)
        else:
            self.logger = get_logger(self.__class__.__name__, lv=logging.ERROR)
        self.init_models(**kwargs)

    def init_models(self, **kwargs):
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float16)
        self.FOCAL_LENGTH = 5000
        self.IMAGE_SIZE = 256
        self.WILOR_MINI_REPO_ID = kwargs.get("WILOR_MINI_REPO_ID", "warmshao/WiLoR-mini")
        wilor_pretrained_dir = kwargs.get("wilor_pretrained_dir",
                                          os.path.join(os.path.dirname(__file__), ".."))
        os.makedirs(wilor_pretrained_dir, exist_ok=True)
        mano_mean_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "mano_mean_params.npz")
        if not os.path.exists(mano_mean_path):
            self.logger.info(f"download mano mean npz {mano_mean_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models",
                            filename="mano_mean_params.npz",
                            local_dir=wilor_pretrained_dir)
        mano_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "MANO_RIGHT.pkl")
        if not os.path.exists(mano_model_path):
            self.logger.info(f"download mano model {mano_model_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="MANO_RIGHT.pkl",
                            local_dir=wilor_pretrained_dir)
        self.logger.info(f"loading WiLor model >>> ")
        self.wilor_model = WiLor(mano_model_path=mano_model_path, mano_mean_path=mano_mean_path,
                                 focal_length=self.FOCAL_LENGTH,
                                 image_size=self.IMAGE_SIZE)
        wilor_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "wilor_final.ckpt")
        if not os.path.exists(wilor_model_path):
            self.logger.info(f"download wilor pretrained model {wilor_model_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="wilor_final.ckpt",
                            local_dir=wilor_pretrained_dir)
        self.wilor_model.load_state_dict(torch.load(wilor_model_path)["state_dict"], strict=False)
        self.wilor_model.eval().to(self.device, dtype=self.dtype)

        yolo_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "detector.pt")
        if not os.path.exists(yolo_model_path):
            self.logger.info(f"download yolo pretrained model {wilor_model_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="detector.pt",
                            local_dir=wilor_pretrained_dir)
        self.logger.info(f"loading Yolo hand detection model >>> ")
        self.hand_detector = YOLO(yolo_model_path)
        self.hand_detector.to(self.device)

        self.IMAGE_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.IMAGE_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

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
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = img_patch_cv / 255.
            img_patch = (img_patch - self.IMAGE_MEAN) / self.IMAGE_STD
            img_patch = np.transpose(img_patch, (2, 0, 1))
            img_patch = torch.from_numpy(img_patch[None]).to(device=self.device, dtype=self.dtype)
            wilor_output = self.wilor_model(img_patch)
            wilor_output = {k: v.cpu().numpy().astype(np.float32) for k, v in wilor_output.items()}
            pred_cam = wilor_output["pred_cam"]
            multiplier = (2 * right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(pred_cam, box_center[None], bbox_size, img_size[None],
                                                     scaled_focal_length)
            wilor_output["pred_cam_t_full"] = pred_cam_t_full
            wilor_output["scaled_focal_length"] = scaled_focal_length
            detect_rets[i]["wilor_preds"] = wilor_output

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
            img_patch_cv = img_patch_cv[:, :, ::-1]
            img_patch = img_patch_cv / 255.
            img_patch = (img_patch - self.IMAGE_MEAN) / self.IMAGE_STD
            img_patch = np.transpose(img_patch, (2, 0, 1))
            img_patch = torch.from_numpy(img_patch[None]).to(device=self.device, dtype=self.dtype)
            wilor_output = self.wilor_model(img_patch)
            wilor_output = {k: v.cpu().numpy().astype(np.float32) for k, v in wilor_output.items()}
            pred_cam = wilor_output["pred_cam"]
            multiplier = (2 * right - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            scaled_focal_length = self.FOCAL_LENGTH / self.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = utils.cam_crop_to_full(pred_cam, box_center[None], bbox_size, img_size[None],
                                                     scaled_focal_length)
            wilor_output["pred_cam_t_full"] = pred_cam_t_full
            wilor_output["scaled_focal_length"] = scaled_focal_length
            detect_rets[i]["wilor_preds"] = wilor_output

        self.logger.info("finish detection!")
        return detect_rets
