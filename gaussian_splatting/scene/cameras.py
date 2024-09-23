#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix2, getProjectionMatrix, fov2focal, focal2fov, getWorld2View2_GS

from utils.pose_utils import SO3_exp


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        image_height,
        image_width,
        R, T,
        fx,
        fy,
        cx,
        cy,
        fovx = None,
        fovy = None,
        kappa = 0.0,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        gt_alpha_mask = None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.R = torch.from_numpy(R)
        self.T = torch.from_numpy(T)

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.kappa = kappa
        self.aspect_ratio = self.fy / self.fx

        if fovx is not None:
            self.FoVx = fovx
        else:
            self.FoVx = focal2fov(fx, image_width)
        if fovy is not None:
            self.FoVy = fovy
        else:
            self.FoVy = focal2fov(fy, image_height)


        self.image_height = image_height
        self.image_width = image_width

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # optimizable parameters
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.cam_focal_delta = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.cam_kappa_delta = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        # old interface property reserved for original 3DGS paper
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(device)

        try:
            self.data_device = torch.device(device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.calibration_identifier = 0


        # backup
        self.R_gt = R
        self.T_gt = T

        self.fx_init = self.fx
        self.fy_init = self.fy
        self.kappa_init = self.kappa


        # add some noise and see if the algorithm can correctly optimize it
        # noiseDeltaR = SO3_exp(torch.tensor([0.01, 0.01, 0.01])).double() 
        # noiseDeltaT = torch.tensor([0.1, 0.1, 0.1], dtype=torch.double)

        # self.T = self.T + noiseDeltaT
        # self.R = self.R @ noiseDeltaR

        


    @property
    def projection_matrix(self):
        return getProjectionMatrix2(
                    znear=self.znear,
                    zfar=self.zfar,
                    fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy,
                    W=self.image_width,
                    H=self.image_height,
                ).transpose(0, 1).to(device=self.device)

    @property
    def world_view_transform(self):
        T = getWorld2View2(self.R, self.T, torch.from_numpy(self.trans), self.scale).transpose(0, 1).to(device=self.device)
        return T

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def update_focal_kappa (self, focal, kappa):
        self.fx = focal
        self.fy = self.aspect_ratio * focal
        self.kappa = kappa

    def update_calibration (self, fx, fy, kappa):
        self.fx = fx
        self.fy = fy
        self.kappa = kappa



    @staticmethod
    def init_from_fov (colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        image_width = image.shape[2]
        image_height = image.shape[1]
        fx = fov2focal(FoVx, image_width)
        fy = fov2focal(FoVy, image_height)
        cx = (image_width - 1) * 0.5
        cy = (image_height - 1) * 0.5
        return Camera(
            uid = colmap_id,
            color = image,
            depth = None,
            image_height = image_height,
            image_width = image_width,
            R = R, T = T,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            fovx = FoVx,
            fovy = FoVy,
            kappa = 0.0,
            trans=trans,
            scale = scale,
            gt_alpha_mask = gt_alpha_mask,
            device=data_device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        img = torch.randint(0, 256, size=(3, H, W), dtype=torch.uint8)
        return Camera (
            uid = uid,
            color = img,
            depth = None,
            image_height = H,
            image_width = W,
            R = T[:3, :3].cpu().numpy(), 
            T = T[:3, 3].cpu().numpy(),
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            fovx = FoVx,
            fovy = FoVy,
            kappa = 0.0,
            trans=np.array([0.0, 0.0, 0.0]),
            scale=1.0,
            gt_alpha_mask = None,
            device="cuda:0",
        )







class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

