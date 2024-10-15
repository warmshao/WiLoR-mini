# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: wilor_hand_pose3d_estimation_pipeline.py
import torch
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

from ..models.wilor import WiLor


class WiLorHandPose3dEstimationPipeline:
    def __init__(self, **kwargs):
        self.init_models(**kwargs)

    def init_models(self, **kwargs):
        self.WILOR_MINI_REPO_ID = kwargs.get("WILOR_MINI_REPO_ID", "warmshao/WiLoR-mini")
        wilor_pretrained_dir = kwargs.get("wilor_pretrained_dir",
                                          os.path.join(os.path.dirname(__file__), ".."))
        os.makedirs(wilor_pretrained_dir, exist_ok=True)
        mano_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "MANO_RIGHT.pkl")
        if not os.path.exists(mano_model_path):
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="MANO_RIGHT.pkl",
                            local_dir=wilor_pretrained_dir)

        self.wilor_model = WiLor(mano_model_path=mano_model_path)
        wilor_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "wilor_final.ckpt")
        if not os.path.exists(wilor_model_path):
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="wilor_final.ckpt",
                            local_dir=wilor_pretrained_dir)
        self.wilor_model.load_state_dict(torch.load(wilor_model_path), strict=False)
        yolo_model_path = os.path.join(wilor_pretrained_dir, "pretrained_models", "detector.pt")
        if not os.path.exists(yolo_model_path):
            hf_hub_download(repo_id=self.WILOR_MINI_REPO_ID, subfolder="pretrained_models", filename="detector.pt",
                            local_dir=wilor_pretrained_dir)
        self.hand_detector = YOLO(yolo_model_path)

    def predict(self, image, **kwargs):
        pass
