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
from ..utils import utils, onnx2trt


class WiLorHandPose3dEstimationPipeline:
    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", True)
        if self.verbose:
            self.logger = get_logger(self.__class__.__name__, lv=logging.INFO)
        else:
            self.logger = get_logger(self.__class__.__name__, lv=logging.ERROR)
        self.init_models(**kwargs)

    def init_models(self, **kwargs):
        # default tot use CPU
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
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
        use_vit_trt = kwargs.get("use_vit_trt", False)
        if use_vit_trt:
            self.logger.info("Use Trt model to speed up >>>")
            vit_trt_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "wilor_vit.trt")
            if not os.path.exists(vit_trt_path):
                vit_onnx_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "wilor_vit.onnx")
                self.logger.info(f"download vit trt model {vit_trt_path} from huggingface")
                hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models",
                                filename="wilor_vit.onnx",
                                local_dir=wilor_pretrained_dir)
                onnx2trt.convert_onnx_to_trt(vit_onnx_path, vit_trt_path, verbose=self.verbose)
        else:
            vit_trt_path = None
        self.logger.info(f"loading WiLor model >>> ")
        self.wilor_model = WiLor(mano_model_path=mano_model_path, mano_mean_path=mano_mean_path,
                                 focal_length=self.FOCAL_LENGTH,
                                 image_size=self.IMAGE_SIZE,
                                 vit_trt_path=vit_trt_path,
                                 **kwargs)
        wilor_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "wilor_final.ckpt")
        if not os.path.exists(wilor_model_path):
            self.logger.info(f"download wilor pretrained model {wilor_model_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="wilor_final.ckpt",
                            local_dir=wilor_pretrained_dir)
        self.wilor_model.load_state_dict(torch.load(wilor_model_path)["state_dict"], strict=False)
        self.wilor_model.eval()
        self.wilor_model.to(self.device, dtype=self.dtype)

        yolo_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "detector.pt")
        if not os.path.exists(yolo_model_path):
            self.logger.info(f"download yolo pretrained model {wilor_model_path} from huggingface")
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="detector.pt",
                            local_dir=wilor_pretrained_dir)
        self.logger.info(f"loading Yolo hand detection model >>> ")
        self.hand_detector = YOLO(yolo_model_path)
        self.hand_detector.to(self.device)

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
            # 弱透视
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
            # 弱透视
            pred_keypoints_2d = utils.perspective_projection(wilor_output_i["pred_keypoints_3d"],
                                                             translation=pred_cam_t_full,
                                                             focal_length=np.array([scaled_focal_length] * 2)[None],
                                                             camera_center=img_size[None] / 2)
            wilor_output_i["pred_keypoints_2d"] = pred_keypoints_2d
            detect_rets[i]["wilor_preds"] = wilor_output_i

        self.logger.info("finish detection!")
        return detect_rets
