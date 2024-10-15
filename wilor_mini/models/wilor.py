import os

import torch
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
        self.backbone = vit()
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

    def forward(self, x):
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
        temp_mano_output = self.mano(**{k: v.float() for k, v in temp_mano_params.items()}, pose2rot=False)
        temp_vertices = temp_mano_output.vertices

        pred_hand_pose, pred_betas, pred_cam = self.refine_net(vit_out, temp_vertices, pred_cam, pred_mano_feats,
                                                               focal_length)

        return pred_hand_pose, pred_betas, pred_cam
