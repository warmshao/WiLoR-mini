# -*- coding: utf-8 -*-
# @Time    : 2024/11/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : AnimateMaster
# @FileName: wilor_hand_pose3d_model.py
import pdb

import numpy as np
import cv2
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict, get_predictor
from ..models.vit import rot6d_to_rotmat


class WilorVitModel:
    """
    WiLoR VIT Model
    """

    def __init__(self, **kwargs):
        self.predictor = get_predictor(**kwargs)
        self.device = torch.cuda.current_device()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        if self.predictor is not None:
            self.input_shapes = self.predictor.input_spec()
            self.output_shapes = self.predictor.output_spec()
        self.NUM_HAND_JOINTS = 15

    def predict_trt(self, *data, **kwargs):
        nvtx.range_push("forward")
        batch_size = data[0].shape[0]
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i].to(device=self.device,
                                                    dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outputs = {}
        for i, out in enumerate(self.predictor.outputs):
            out_shape = out["shape"]
            out_shape[0] = batch_size
            out_tensor = preds_dict[out["name"]][:np.prod(out_shape)].reshape(*out_shape)
            outputs[out["name"]] = out_tensor
        nvtx.range_pop()
        return outputs

    def __call__(self, *data, **kwargs):
        image = data[0]
        outputs = self.predict_trt(image, **kwargs)
        pred_hand_pose, pred_betas, pred_cam, img_feat = outputs["pred_hand_pose"], outputs["pred_betas"], outputs[
            "pred_cam"], outputs["img_feat"]
        B = pred_hand_pose.shape[0]
        pred_mano_feats = {}
        pred_mano_feats['hand_pose'] = pred_hand_pose
        pred_mano_feats['betas'] = pred_betas
        pred_mano_feats['cam'] = pred_cam
        pred_hand_pose = rot6d_to_rotmat(pred_hand_pose).view(B, self.NUM_HAND_JOINTS + 1, 3, 3)
        pred_mano_params = {'global_orient': pred_hand_pose[:, :1],
                            'hand_pose': pred_hand_pose[:, 1:],
                            'betas': pred_betas}
        return pred_mano_params, pred_cam, pred_mano_feats, img_feat
