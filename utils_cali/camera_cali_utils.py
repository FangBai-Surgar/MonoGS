import torch
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask
from utils.camera_utils import Camera

class CameraForCalibration(Camera):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        # projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        cali_id=None,
        device="cuda:0"
    ):
        super().__init__(
            uid,
            color,
            depth,
            gt_T,
            fx,
            fy,
            cx,
            cy,
            fovx,
            fovy,
            image_height,
            image_width,
            device,
        )
        if cali_id is None:
            self.calibration_identifier = 0
        else:
            self.calibration_identifier = cali_id


    @staticmethod
    def init_from_dataset(dataset, idx):
        if dataset.focal_changed: # property of the simulated dataset
            gt_color, gt_depth, gt_pose, fx, fy, cx, cy, fovx, fovy, height, width, cali_id = dataset[idx]
            return CameraForCalibration(
                idx,
                gt_color,
                gt_depth,
                gt_pose,
                fx,
                fy,
                cx,
                cy,
                fovx,
                fovy,
                height,
                width,
                cali_id,
                device=dataset.device,
            )
        else:
            gt_color, gt_depth, gt_pose = dataset[idx] 
            return CameraForCalibration(
                idx,
                gt_color,
                gt_depth,
                gt_pose,
                # projection_matrix,
                dataset.fx,
                dataset.fy,
                dataset.cx,
                dataset.cy,
                dataset.fovx,
                dataset.fovy,
                dataset.height,
                dataset.width,
                None,
                device=dataset.device,
            )
    
    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        # projection_matrix = getProjectionMatrix2(
        #     znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        # ).transpose(0, 1)
        return CameraForCalibration(
            uid, None, None, T, fx, fy, cx, cy, FoVx, FoVy, H, W
        )