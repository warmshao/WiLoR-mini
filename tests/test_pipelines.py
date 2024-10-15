# -*- coding: utf-8 -*-
# @Time    : 2024/10/14
# @Author  : wenshao
# @Project : WiLoR-mini
# @FileName: test_wilor_pipeline.py

def test_wilor_pipeline():
    import cv2
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    pipe = WiLorHandPose3dEstimationPipeline()
    img_path = "assets/img.png"
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    test_wilor_pipeline()
