import os
import pdb
import time
import numpy as np
import torch
import roma
from torch import nn
from .vit import vit
from .refinement_net import RefineNet
from .mano_wrapper import MANO


class WiLor(nn.Module):
    """
    WiLor for Onnx
    """

    def __init__(self, **kwargs):
        super(WiLor, self).__init__()
        # Create VIT backbone
        self.backbone = vit(**kwargs)
        # Create RefineNet head
        self.refine_net = RefineNet(feat_dim=1280, upscale=3)
        mano_model_path = kwargs.get("mano_model_path", "")
        assert os.path.exists(mano_model_path), f"MANO model {mano_model_path} not exists!"
        mano_cfg = {
            "model_path": mano_model_path,
            "create_body_pose": False
        }
        self.mano = MANO(**mano_cfg)
        self.FOCAL_LENGTH = kwargs.get("focal_length", 5000)
        self.IMAGE_SIZE = kwargs.get("image_size", 256)
        self.IMAGE_MEAN = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3))
        self.IMAGE_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225])).reshape(1, 1, 1, 3)

    def forward(self, x):
        x = x.flip(dims=[-1]) / 255.0
        x = (x - self.IMAGE_MEAN.to(x.device, dtype=x.dtype)) / self.IMAGE_STD.to(x.device, dtype=x.dtype)
        x = x.permute(0, 3, 1, 2)
        batch_size = x.shape[0]
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(x[:, :, :, 32:-32])  # B, 1280, 16, 12
        # Compute camera translation
        focal_length = self.FOCAL_LENGTH * torch.ones(batch_size, 2, device=x.device, dtype=x.dtype)

        ## Temp MANO
        temp_mano_params['global_orient'] = temp_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['hand_pose'] = temp_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        temp_mano_params['betas'] = temp_mano_params['betas'].reshape(batch_size, -1)
        temp_mano_output = self.mano(**temp_mano_params, pose2rot=False)
        temp_vertices = temp_mano_output.vertices

        pred_mano_params = self.refine_net(vit_out, temp_vertices, pred_cam, pred_mano_feats,
                                           focal_length)
        mano_output = self.mano(**pred_mano_params, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices

        pred_mano_params['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        pred_mano_params['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_mano_params['global_orient'] = roma.rotmat_to_rotvec(pred_mano_params['global_orient'])
        pred_mano_params['hand_pose'] = roma.rotmat_to_rotvec(pred_mano_params['hand_pose'])
        return pred_mano_params
