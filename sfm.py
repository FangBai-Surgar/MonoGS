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


import os
import sys
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
import uuid
from tqdm import tqdm
import wandb
from random import randint
import numpy as np
import copy

import torch
import torch.multiprocessing as mp
import torch.optim.lr_scheduler as lr_scheduler


from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.utils.general_utils import safe_state
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.utils.graphics_utils import BasicPointCloud

from gaussian_splatting.utils.general_utils import helper as lr_helper


from utils.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from utils.multiprocessing_utils import FakeQueue, clone_obj


from depth_anything import DepthAnything


from optimizers import CalibrationOptimizer, PoseOptimizer, LineDetection, lr_exp_decay_helper

from gaussian_scale_space import image_conv_gaussian_separable

from matplotlib import pyplot as plt

import pathlib
import rich


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False







class SFM(mp.Process):


    def __init__(self, pipe = None, q_main2vis = None, q_vis2main = None, use_gui = True, viewpoint_stack = None, gaussians = None, opt = None, cameras_extent = None) -> None:
        self.pipe = pipe
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.use_gui = use_gui

        self.viewpoint_stack = viewpoint_stack
        self.gaussians = gaussians
        self.opt = opt

        self.dense_point_cloud = None
        self.add_dense_pcd_iter = -1

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01

        self.pause = False
        

        self.require_calibration = True
        self.allow_lens_distortion = True

        self.focal_reference = None

        self.start_calib_iter = 50
        self.stop_calib_iter = 300

        self.start_pose_iter = 50
        self.stop_pose_iter = 300

        self.start_gaussian_iter = 0
        self.stop_gaussian_iter = 100000


        self.cameras_extent = cameras_extent
        self.depth_anything = DepthAnything()

        self.calibration_optimizer = None
        self.pose_optimizer = None
        self.calib_safe_guard = False

        self.gaussian_scale_t = 10
        self.image_margin_mask = None


        self.MODULE_TEST_CALIBRATION = False
        self.add_calib_noise_iter = -1


    def push_to_gui (self, cam_cnt):
        # depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))

        # use depth prediction from a Neural network                    
        cv_img = (self.viewpoint_stack[cam_cnt].original_image*255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        depth = self.depth_anything.eval(cv_img)

        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=self.gaussians,
                keyframes=copy.deepcopy(self.viewpoint_stack),
                current_frame=clone_obj(self.viewpoint_stack[cam_cnt]),
                gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                gtdepth=depth,
            )
        )
        time.sleep(0.001)



    def compute_loss (self, frozen_states = False, use_SSIM = False):
        # Loss function
        loss = 0.0

        self.viewspace_point_tensor_acm = []
        self.visibility_filter_acm = []
        self.radii_acm = []
        self.n_touched_acm = []

        for viewpoint in self.viewpoint_stack:

            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)

            image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]                   

            # Loss
            gt_image = viewpoint.original_image.cuda() 
            mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold)
            # mask = opacity
            # Ll1 = l1_loss(image, gt_image)

            # Gaussian scale space for focal length calibration
            if frozen_states and self.gaussian_scale_t > 0.5:
                image_scale_t = image_conv_gaussian_separable(image, sigma=self.gaussian_scale_t, epsilon=0.01)
                gt_image_scale_t = image_conv_gaussian_separable(gt_image, sigma=self.gaussian_scale_t, epsilon=0.01)
                # mask the image margin when peforming long range pixel matching
                mask = mask * self.image_margin_mask
            else:
                image_scale_t = image
                gt_image_scale_t = gt_image

            # huber_loss_function = torch.nn.HuberLoss(reduction = 'mean', delta = 1.0)
            huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = 0.0)
            loss += (1.0 - self.opt.lambda_dssim) * huber_loss_function(image_scale_t*mask, gt_image_scale_t*mask)

            # Ll1 = l1_loss(image*mask, gt_image*mask)  
            # loss += (1.0 - self.opt.lambda_dssim) * Ll1

            # enable SSIM loss when a good intialial reconstruction is attained
            if use_SSIM:
                loss += self.opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))

            self.viewspace_point_tensor_acm.append(viewspace_point_tensor)
            self.visibility_filter_acm.append(visibility_filter)
            self.radii_acm.append(radii)
            self.n_touched_acm.append(n_touched)


        return loss
    



    def set_gaussian_densification_stats (self):
        for idx in range(len(self.viewspace_point_tensor_acm)):
            self.gaussians.max_radii2D[self.visibility_filter_acm[idx]] = torch.max(
                self.gaussians.max_radii2D[self.visibility_filter_acm[idx]],
                self.radii_acm[idx][self.visibility_filter_acm[idx]],
            )
            self.gaussians.add_densification_stats(
                self.viewspace_point_tensor_acm[idx], self.visibility_filter_acm[idx]
            )


    def add_dense_point_cloud (self, positions, colors, normals = None):
        if positions is None or colors is None:
            self.dense_point_cloud = None
            return        
        self.dense_point_cloud = BasicPointCloud(points=positions, colors=colors, normals=normals)
        

    def optimize (self):


        _, h, w = self.viewpoint_stack[0].original_image.shape
        self.image_margin_mask = torch.zeros(h, w).cuda()
        band_with = int(1.0 * self.gaussian_scale_t)
        self.image_margin_mask[band_with:-band_with,  band_with:-band_with] = 1.0
        if self.focal_reference is None:
            self.focal_reference = np.sqrt(h*h + w*w)/2


        if self.calibration_optimizer is None:            
            self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            self.calibration_optimizer.update_focal_learning_rate (lr = 0.1)
            self.calib_safe_guard = False


        if self.pose_optimizer is None:
            self.pose_optimizer = PoseOptimizer(self.viewpoint_stack)



        cam_cnt = 0
        if self.use_gui:
            self.push_to_gui(cam_cnt)
            time.sleep(1.5)


        sfm_gui.Log("start SfM optimization")

        first_iter = 0

        self.gaussians.training_setup(self.opt)

   
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)


        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1


        loss_prev = 1e10
        undo_prev = False


        for iteration in range(first_iter, self.opt.iterations+1):
            

            if (self.dense_point_cloud is not None) and (iteration == self.add_dense_pcd_iter):
                self.gaussians.extend_from_pcd(self.dense_point_cloud, point_size=1.0)
                self.gaussians.training_setup(self.opt)
                sfm_gui.Log(f"Include additional dense point cloud: {len(self.dense_point_cloud.points)} points", tag="SFM")


            # interaction with gui interface Pause/Resume
            if not self.q_vis2main.empty():
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause            
                while self.pause:
                    if self.q_vis2main.empty():
                            time.sleep(0.01)
                            continue
                    else:
                        data_vis2main = self.q_vis2main.get()
                        self.pause = data_vis2main.flag_pause



            iter_start.record()


            # add noise to calibration to test the robustness
            if self.MODULE_TEST_CALIBRATION and iteration == self.add_calib_noise_iter:
                for viewpoint_cam in self.viewpoint_stack:
                    focal = 400 # gt = 580. tunning in range [400 - 700]
                    viewpoint_cam.fx = focal
                    viewpoint_cam.fy = viewpoint_cam.aspect_ratio * focal
                    viewpoint_cam.kappa = 0.0

                if self.use_gui:
                    cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
                    self.push_to_gui(cam_cnt)
                time.sleep(3)


            frozen_states = ( iteration >= self.start_calib_iter and iteration < self.start_calib_iter + 50 )


            if iteration == self.start_calib_iter + 50: # change to SGD
                lr = self.calibration_optimizer.estimate_step_size()
                self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "SGD")
                self.calibration_optimizer.update_focal_learning_rate (lr = lr)
                self.calib_safe_guard = True
            

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # FORWARD
            loss = self.compute_loss (
                    frozen_states = frozen_states,
                    use_SSIM = (iteration > self.stop_calib_iter) )


            iter_end.record()

            # calibration step safe-guard
            if self.require_calibration and iteration >= self.start_calib_iter and iteration < self.stop_calib_iter:
                if self.calib_safe_guard and loss > loss_prev:
                    rich.print(f"[bold yellow][Warning]: learning rate is too big! revoke previous step [/bold yellow]")
                    # print(f"loss_prev = {loss_prev},   loss = {loss},   current_focal = {self.viewpoint_stack[0].fx}")
                    self.calibration_optimizer.undo_focal_step()
                    # print(f"\t after revoking, current_fx = {self.viewpoint_stack[0].fx}")

                    if (not undo_prev):
                        undo_prev = True
                        # recompute the loss, as Gaussian/pose update may have changed it
                        loss_prev = self.compute_loss (
                                frozen_states = frozen_states,
                                use_SSIM = (iteration > self.stop_calib_iter) )
                    # verify the loss again
                    if loss > loss_prev:
                        self.calibration_optimizer.update_focal_learning_rate (scale = 0.5)

                    with torch.no_grad():
                        self.calibration_optimizer.focal_step() # focal step with old gradient
                    # print(f"\t update to     , current_fx = {self.viewpoint_stack[0].fx}\n")
                    continue


            undo_prev = False
            if loss < loss_prev:
                loss_prev = loss


            self.calibration_optimizer.zero_grad() # clear gradient every iteration
            self.pose_optimizer.zero_grad() # clear gradient every iteration
            self.gaussians.optimizer.zero_grad(set_to_none = True) # clear gradient every iteration

            loss.backward()


            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Densification
                if True and iteration < self.opt.densify_until_iter:

                    self.set_gaussian_densification_stats()

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                    
                    if iteration % self.opt.opacity_reset_interval == 0:
                        sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                        self.gaussians.reset_opacity()


                # calibration step
                if self.require_calibration and iteration >= self.start_calib_iter and iteration < self.stop_calib_iter:
                    self.calibration_optimizer.focal_step()
                    if not frozen_states and self.allow_lens_distortion:
                        self.calibration_optimizer.kappa_step()
                
                # pose step
                if iteration >= self.start_pose_iter and iteration < self.stop_pose_iter:
                    self.pose_optimizer.step()
                

                # Gaussian step
                if iteration >= self.start_gaussian_iter and iteration < self.stop_gaussian_iter:
                    self.gaussians.optimizer.step()


                if self.use_gui and (iteration % 10 == 0 or frozen_states):
                    cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
                    self.push_to_gui(cam_cnt)
                    if frozen_states:
                        time.sleep(0.05)



        focal_stack, focal_grad_stack = self.calibration_optimizer.get_focal_statistics(all = True)
        LineDetection(focal_stack[:80], focal_grad_stack[:80]).plot_figure(fname = pathlib.Path.home()/( "focal_cost_function_scale"+str(self.gaussian_scale_t)+".pdf" ) )
         
        




        sfm_gui.Log(f"SfM optimization complete with {iteration} iterations.")

        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))  
            time.sleep(3.0)







if __name__ == "__main__":

    mp.set_start_method('spawn')


    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)


    # print(opt.__dict__)
    # print(pipe.__dict__)



    opt.iterations = 200
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 750
    opt.densify_grad_threshold = 0.0002



    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    cameras_extent = scene.cameras_extent


    N = 3

    viewpoint_stack = scene.getTrainCameras()
    while len(viewpoint_stack) > N:
        viewpoint_stack.pop(-1)
    sfm_gui.Log(f"cameras used: {len(scene.getTrainCameras())}")

    viewpoint_stack = scene.getTrainCameras().copy()

    # in original 3DGS, R is transposed in colmap reader and later inverted in getWorld2View2
    # in this code, getWorld2View2 don't transpose R
    for cam in viewpoint_stack:
        Rt = torch.transpose(cam.R, 0, 1)
        cam.R = Rt



    torch.autograd.set_detect_anomaly(args.detect_anomaly)



    ## visualization
    use_gui = True
    q_main2vis = mp.Queue() if use_gui else FakeQueue()
    q_vis2main = mp.Queue() if use_gui else FakeQueue()


    if use_gui:
        bg_color = [0.0, 0.0, 0.0]
        params_gui = gui_utils.ParamsGUI(
            pipe=pipe,
            background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            gaussians=GaussianModel(dataset.sh_degree),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(1)


    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    sfm = SFM(pipe, q_main2vis, q_vis2main, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)

    sfm.MODULE_TEST_CALIBRATION = True
    sfm.add_calib_noise_iter = 50
    sfm.start_calib_iter = 50

    sfm_process = mp.Process(target=sfm.optimize)
    sfm_process.start()

  
    torch.cuda.synchronize()


    if use_gui:
        gui_process.join()
        sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")



    sfm_process.join()
    sfm_gui.Log("Finished", tag="SfM")
