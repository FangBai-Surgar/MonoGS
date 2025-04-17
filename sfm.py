
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
import random

import torch
torch.cuda.set_device(0)
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


from slam_impl.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from slam_impl.multiprocessing_utils import FakeQueue, clone_obj


# from utils.depth_anything import DepthAnything


from slam_impl.optimizers import CalibrationOptimizer, PoseOptimizer, LineDetection, lr_exp_decay_helper

from slam_impl.gaussian_scale_space import image_conv_gaussian

from matplotlib import pyplot as plt

import pathlib
import rich

from gaussian_viewer import Viewer, create_gaussians_gl


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False





def print_viewpoint_stack(viewpoint_stack, prefix="Camera"):
    for viewpoint_cam in viewpoint_stack:
        uid = viewpoint_cam.uid
        calib_id = viewpoint_cam.calib_id
        fx = viewpoint_cam.fx
        fy = viewpoint_cam.fy
        kappa = viewpoint_cam.kappa     
        CC = viewpoint_cam.camera_center.cpu().numpy()
        exposure_a = viewpoint_cam.exposure_a.data.item()
        exposure_b = viewpoint_cam.exposure_b.data.item()
        rich.print(f"[bold blue]{prefix}[/bold blue] uid: [{uid:05d}]: calib_id: {calib_id}. fx: {fx:.3f}, fy: {fy:.3f}, kappa: {kappa:.6f}. cam_center: ({CC[0]:.3f}, {CC[1]:.3f}, {CC[2]:.3f}), exposure: (a: {exposure_a:.5f}, b: {exposure_b:.5f})")



class SFM(mp.Process):


    def __init__(self, pipe = None, use_gui = True, viewpoint_stack = None, gaussians = None, opt = None, cameras_extent = None) -> None:
        self.pipe = pipe
        self.use_gui = use_gui

        self.viewpoint_stack = viewpoint_stack
        self.gaussians = gaussians
        self.opt = opt

        self.gaussian_iter = 0
        self.verbose = False

        self.dense_point_cloud = None
        self.add_dense_pcd_iter = -1

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01

        self.pause = False
        

        self.require_calibration = self.opt.require_calibration
        self.allow_lens_distortion = self.opt.allow_lens_distortion

        self.focal_reference = None
        self.cameras_extent = cameras_extent
        # self.depth_anything = DepthAnything()

        self.calibration_optimizer = None
        self.pose_optimizer = None
        self.calib_safe_guard = False

        self.gaussian_scale_t = 10
        self.image_margin_mask = None


        self.MODULE_TEST_CALIBRATION = False
        self.add_calib_noise_iter = -1
        

        self.q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        self.q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        if self.use_gui:
            bg_color = [0.0, 0.0, 0.0]
            params_gui = gui_utils.ParamsGUI(
                pipe=pipe,
                background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            self.gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
            self.gui_process.start()
            time.sleep(3)


    """
    main function
    """
    def optimize (self, phase1_iter = 200, phase3_iter = 500, phase2_DBA_iter = 100, phase2_CaliDBA_iter = 500, phase2_CaliDBA_GSS_iter = 0, use_scale_space = True, set_focal_error=None, viewpoint_stack_reset=None):

        _, h, w = self.viewpoint_stack[0].original_image.shape
        self.image_margin_mask = torch.zeros(h, w).cuda()
        band_with = int(1.0 * self.gaussian_scale_t)
        self.image_margin_mask[band_with:-band_with,  band_with:-band_with] = 1.0
        if self.focal_reference is None:
            self.focal_reference = np.sqrt(h*h + w*w)/2

        if self.calibration_optimizer is None:
            self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            self.calib_safe_guard = False

        if self.pose_optimizer is None:
            self.pose_optimizer = PoseOptimizer(self.viewpoint_stack)

        cam_cnt = 0
        if self.use_gui:
            self.push_to_gui(cam_cnt)
            time.sleep(1.5)

        self.gaussians.training_setup(self.opt)

        sfm_gui.Log("start SfM optimization")
        if self.verbose:
            print_viewpoint_stack(self.viewpoint_stack)

        # Gaussian initialization
        self.run_phase1(max_iters = phase1_iter)

        # Bundle adjustment (no calibration, no pose update) to densify 3D Gaussians
        self.run_phase2(max_iters = phase2_DBA_iter, update_Gaussian = True, update_pose = False, update_calibration = False, use_scale_space = False)

        if set_focal_error is not None:
            ''' For debug and test
            '''
            noise_fx = set_focal_error
            for viewpoint in self.viewpoint_stack:
                focal = viewpoint.fx + noise_fx
                viewpoint.fx = focal
                viewpoint.fy = viewpoint.aspect_ratio * focal
            if self.verbose:
                rich.print(f"[bold red][Notice]: old fx {focal - noise_fx} ====> new fx {focal}.  Noise added {noise_fx}  [/bold red]")

        if viewpoint_stack_reset is not None:
            for cam in self.viewpoint_stack:
                cam.clean()
            self.viewpoint_stack = None
            torch.cuda.empty_cache()
            self.viewpoint_stack = viewpoint_stack_reset

        # 
        """
        SelfCalibrating Bundle adjustment

        Note: in the guassian scale space optimization, make sure fix 3D Gaussians.
        """
        self.run_phase2(max_iters = phase2_CaliDBA_GSS_iter, update_Gaussian = False, update_pose = False, update_calibration = True, use_scale_space = use_scale_space)
        self.run_phase2(max_iters = phase2_CaliDBA_GSS_iter, update_Gaussian = False, update_pose = True,  update_calibration = True, use_scale_space = False)
        self.run_phase2(max_iters = phase2_CaliDBA_iter,     update_Gaussian = True,  update_pose = True,  update_calibration = True, use_scale_space = False)
        
        # refinement using SSIM 
        self.run_phase3(max_iters = phase3_iter)

        # self.show_rendered_images()

        sfm_gui.Log(f"SfM optimization complete.")
        torch.cuda.synchronize()

        self.close()


    """
    optimization implementation
    """

    def run_phase1 (self, max_iters = 500):
        '''
        Initialize 3D Gaussians for sparse SfM point-cloud
        '''
        if (max_iters <=0):
            return
        progress_bar = tqdm(range(1, max_iters+1), desc="Phase1: Training progress")
        cam_cnt = 0
        for iteration in range(1, max_iters+1):
            self.read_gui_ctrl()            
            self.optimize_one_step_sequel (
                                           use_scale_space = False,
                                           use_ssim_loss = False
                                           )
            if self.use_gui and (iteration % 10 == 0):
                self.push_to_gui(cam_cnt)
                cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
            if iteration % 10 == 0:
                with torch.no_grad():
                    loss_log = self.compute_loss (use_scale_space = False,  use_SSIM = False )
                progress_bar.set_postfix({"Loss": f"{loss_log:.{7}f}"})
                progress_bar.update(10)
        progress_bar.close()


    def run_phase2 (self, max_iters = 1000, update_Gaussian = False, update_pose = False, update_calibration = False, use_scale_space = False):
        '''
        BA (Gaussian, pose, calibration)
        '''
        if (max_iters <=0):
            return
        self.pose_optimizer = PoseOptimizer(self.viewpoint_stack)
        self.pose_optimizer.zero_grad()
        self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
        self.calibration_optimizer.update_focal_learning_rate (lr = 0.02)   if (not update_Gaussian) and (not update_pose) else 0.002
        self.calibration_optimizer.update_kappa_learning_rate (lr = 0.001)  if (not update_Gaussian) and (not update_pose) else 0.001
        self.calibration_optimizer.zero_grad()
        progress_bar = tqdm(range(1, max_iters+1), desc="Phase2: Training progress")
        cam_cnt = 0
        for iteration in range(1, max_iters+1):
            self.read_gui_ctrl()
            (densify_prune, reset_opacity) = self.is_densify_prune()
            reset_opacity =  False
            self.optimize_one_step (iteration,
                                    update_Gaussian = update_Gaussian,
                                    update_pose = update_pose,
                                    update_calibration = update_calibration,
                                    use_scale_space = use_scale_space,
                                    densify_prune = densify_prune,
                                    reset_opacity = reset_opacity
                                    )
            if self.verbose:
                print_viewpoint_stack(self.viewpoint_stack, prefix="Camera")
            if self.use_gui and (iteration % 10 == 0):
                self.push_to_gui(cam_cnt)
                cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
            if iteration % 10 == 0:
                with torch.no_grad():
                    loss_log = self.compute_loss (use_scale_space = False,  use_SSIM = False )
                progress_bar.set_postfix({"Loss": f"{loss_log:.{7}f}"})
                progress_bar.update(10)
        progress_bar.close()


    def run_phase3 (self, max_iters = 500):
        '''
        Refine 3D Gaussians with SSIM loss
        '''
        if (max_iters <=0):
            return
        progress_bar = tqdm(range(1, max_iters+1), desc="Phase3: Training progress")
        cam_cnt = 0
        for iteration in range(1, max_iters+1):
            self.read_gui_ctrl()
            (densify_prune, reset_opacity) = self.is_densify_prune()
            self.optimize_one_step (iteration,
                                    update_Gaussian = True,
                                    use_ssim_loss = True,
                                    densify_prune = densify_prune,
                                    reset_opacity = reset_opacity
                                    )
            if self.use_gui and (iteration % 10 == 0):
                self.push_to_gui(cam_cnt)
                cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
            if iteration % 10 == 0:
                with torch.no_grad():
                    loss_log = self.compute_loss (use_scale_space = False,  use_SSIM = False )
                progress_bar.set_postfix({"Loss": f"{loss_log:.{7}f}"})
                progress_bar.update(10)
        progress_bar.close()


    def is_densify_prune (self):
        densify_prune = ( (self.gaussian_iter > self.opt.densify_from_iter) and (self.gaussian_iter % self.opt.densification_interval ==0) )
        reset_opacity = (self.gaussian_iter % self.opt.opacity_reset_interval ==0)
        return densify_prune, reset_opacity



    """
    Additional Output
    """

    def show_rendered_images (self):
        for viewpoint in self.viewpoint_stack:
            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)
            image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]
            # convert torch tensor to opencv image
            rgb = torch.clamp(image, min=0, max=1.0) * 255
            rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
            plt.imshow(rgb)
            plt.title(f"view uid: {viewpoint.uid}", fontweight ="bold") 
            plt.show()


    def eval_data(self):
        W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images = [], [], [], [], [], [], []
        for viewpoint in self.viewpoint_stack:
            R = viewpoint.R.detach().cpu().numpy()
            T = viewpoint.T.detach().cpu().numpy()
            fx = viewpoint.fx
            fy = viewpoint.fy
            kappa = viewpoint.kappa
            original_image = viewpoint.original_image.detach().cpu()
            rendering = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)["render"].detach().cpu()
            render_image = torch.clamp(rendering, 0.0, 1.0)
            W2C = np.eye(4)
            W2C[0:3, 0:3] = R
            W2C[0:3, 3] = T
            W2C_arr.append (W2C)
            fx_arr.append (fx)
            fy_arr.append (fy)
            kappa_arr.append (kappa)
            rendered_images.append(render_image)
            captured_images.append(original_image)
            error_images.append(torch.abs(rendering - original_image))
            # print(error_images[-1])
        return (W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images) 
    

    @staticmethod
    def tensor2rgb (image):
        '''
            OpenCV stores images in BGR order instead of RGB
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        '''
        rgb = torch.clamp(image, min=0, max=1.0) * 255
        rgb = rgb.byte().permute(1, 2, 0).contiguous().cpu().numpy()
        return rgb
    

    """
    GUI
    """


    def close(self):
        torch.cuda.synchronize()
        if self.use_gui:
            self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            self.gui_process.join()
            sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")
        time.sleep(0.01)


    def push_to_gui (self, cam_cnt):
        depth = np.zeros((self.viewpoint_stack[0].image_height, self.viewpoint_stack[0].image_width))

        # use depth prediction from a Neural network                    
        # cv_img = (self.viewpoint_stack[cam_cnt].original_image*255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        # depth = self.depth_anything.eval(cv_img)

        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                keyframes=self.viewpoint_stack,
                current_frame=self.viewpoint_stack[cam_cnt],
                gtcolor=self.viewpoint_stack[cam_cnt].original_image,
                gtdepth=depth,
            )
        )
        time.sleep(0.001)
    

    def read_gui_ctrl (self):
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


    """
    Loss computation
    """

    def compute_loss_one_view (self, viewpoint, use_scale_space = False, use_smooth_l1 = False, use_SSIM = False):
        # Loss function
        loss = 0.0

        render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                            scaling_modifier=1.0,
                            override_color=None,
                            mask=None,)

        image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]                   

        # Loss
        gt_image = viewpoint.original_image.cuda() 
        mask = (gt_image.sum(dim=0) > self.rgb_boundary_threshold)
        # mask = mask * opacity

        # Gaussian scale space for focal length calibration
        if use_scale_space and self.gaussian_scale_t > 0.5:
            image_scale_t = image_conv_gaussian(image, sigma=self.gaussian_scale_t, epsilon=0.01)
            gt_image_scale_t = image_conv_gaussian(gt_image, sigma=self.gaussian_scale_t, epsilon=0.01)
        else:
            image_scale_t = image
            gt_image_scale_t = gt_image


        if use_smooth_l1:
            """
            Use a Huber-type loss function for smooth gradients at minumum
            - HuberLoss
            - SmoothL1Loss
            parameters decided by residual = |f(x) - y|
            """
            huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = 0.001) # or 0.001, use small enough beta for smooth gradient close to groud-truth
            Ll1 =  huber_loss_function(image_scale_t*mask, gt_image_scale_t*mask)
            loss += (1.0 - self.opt.lambda_dssim) * Ll1 if use_SSIM else Ll1

        else:
            """
            standard L1 loss
            """
            Ll1 = l1_loss(image_scale_t*mask, gt_image_scale_t*mask)
            loss += (1.0 - self.opt.lambda_dssim) * Ll1 if use_SSIM else Ll1

        # enable SSIM loss when a good intialial reconstruction is attained
        if use_SSIM:
            loss += self.opt.lambda_dssim * (1.0 - ssim(image_scale_t*mask, gt_image_scale_t*mask))

        return loss, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched
    


    def compute_loss (self, use_scale_space = False, use_SSIM = False):
        # Loss function
        loss = 0.0

        self.viewspace_point_tensor_acm = []
        self.visibility_filter_acm = []
        self.radii_acm = []
        self.n_touched_acm = []

        for viewpoint in self.viewpoint_stack:

            view_loss, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = self.compute_loss_one_view (viewpoint, use_scale_space = use_scale_space, use_smooth_l1 = False, use_SSIM = use_SSIM)

            loss += view_loss

            self.viewspace_point_tensor_acm.append(viewspace_point_tensor)
            self.visibility_filter_acm.append(visibility_filter)
            self.radii_acm.append(radii)
            self.n_touched_acm.append(n_touched)

        return loss


    def optimize_one_step_sequel (self, use_scale_space = False, use_ssim_loss = False):

        for viewpoint in self.viewpoint_stack:

            self.gaussian_iter += 1
            self.gaussians.update_learning_rate(self.gaussian_iter)

            # FORWARD
            loss, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = self.compute_loss_one_view ( viewpoint, use_scale_space = use_scale_space, use_smooth_l1 = False, use_SSIM = use_ssim_loss )            

            # BACKWARD
            loss.backward()

            with torch.no_grad():

                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                (densify_prune, reset_opacity) = self.is_densify_prune()

                if densify_prune:
                    sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                    size_threshold = 20 if self.gaussian_iter > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                
                if reset_opacity:
                    sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none = True) # clear gradient every iteration

                self.calibration_optimizer.zero_grad() # clear gradient every iteration
                self.pose_optimizer.zero_grad() # clear gradient every iteration


    def optimize_one_step (self, iteration, update_Gaussian = False, update_pose = False, update_calibration = False,  use_scale_space = False, use_ssim_loss = False, densify_prune = False, reset_opacity = False):

        # FORWARD
        loss = self.compute_loss ( use_scale_space = use_scale_space,  use_SSIM = use_ssim_loss )

        # BACKWARD
        loss.backward()

        with torch.no_grad():

            # calibration step
            if update_calibration:
                
                # for viewpoint in self.viewpoint_stack:
                #     rich.print(f"[bold yellow]After loss.backward: [/bold yellow]{viewpoint.cam_focal_delta.grad=}")
                self.calibration_optimizer.focal_step()

                if self.allow_lens_distortion and iteration > 50:
                    # for viewpoint in self.viewpoint_stack:
                    #     rich.print(f"[bold red]After loss.backward: [/bold red]{viewpoint.cam_kappa_delta.grad=}")
                    self.calibration_optimizer.kappa_step()
            
            # pose step
            if update_pose:
                self.pose_optimizer.step(fix_uid = self.viewpoint_stack[0].uid)
            
            # Gaussian step
            if update_Gaussian:

                self.gaussian_iter += 1
                self.gaussians.update_learning_rate(self.gaussian_iter)

                # Gaussian Dynamic Control. Densification and Prune
                self.set_gaussian_densification_stats()

                if densify_prune:
                    sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
                    size_threshold = 20 if self.gaussian_iter > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                
                if reset_opacity:
                    sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()

            self.calibration_optimizer.zero_grad() # clear gradient every iteration
            self.pose_optimizer.zero_grad() # clear gradient every iteration
            self.gaussians.optimizer.zero_grad(set_to_none = True) # clear gradient every iteration



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
        
    

    # def optimize_backup (self, update_Gaussian = False, update_pose = False, update_calibration = False,  use_ssim_loss = False):


    #     _, h, w = self.viewpoint_stack[0].original_image.shape
    #     self.image_margin_mask = torch.zeros(h, w).cuda()
    #     band_with = int(1.0 * self.gaussian_scale_t)
    #     self.image_margin_mask[band_with:-band_with,  band_with:-band_with] = 1.0
    #     if self.focal_reference is None:
    #         self.focal_reference = np.sqrt(h*h + w*w)/2


    #     if self.calibration_optimizer is None:            
    #         self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
    #         self.calibration_optimizer.update_focal_learning_rate (lr = 0.1)
    #         self.calib_safe_guard = False


    #     if self.pose_optimizer is None:
    #         self.pose_optimizer = PoseOptimizer(self.viewpoint_stack)



    #     cam_cnt = 0
    #     if self.use_gui:
    #         self.push_to_gui(cam_cnt)
    #         time.sleep(1.5)


    #     sfm_gui.Log("start SfM optimization")

    #     first_iter = 0

    #     self.gaussians.training_setup(self.opt)

   
    #     iter_start = torch.cuda.Event(enable_timing = True)
    #     iter_end = torch.cuda.Event(enable_timing = True)


    #     ema_loss_for_log = 0.0
    #     progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
    #     first_iter += 1


    #     loss_prev = 1e10
    #     undo_prev = False


    #     for iteration in range(first_iter, self.opt.iterations+1):
            

    #         if (self.dense_point_cloud is not None) and (iteration == self.add_dense_pcd_iter):
    #             self.gaussians.extend_from_pcd(self.dense_point_cloud, point_size=1.0)
    #             self.gaussians.training_setup(self.opt)
    #             sfm_gui.Log(f"Include additional dense point cloud: {len(self.dense_point_cloud.points)} points", tag="SFM")


    #         self.read_gui_ctrl()


    #         iter_start.record()


    #         # add noise to calibration to test the robustness
    #         if self.MODULE_TEST_CALIBRATION and iteration == self.add_calib_noise_iter:
    #             for viewpoint_cam in self.viewpoint_stack:
    #                 focal = 400 # gt = 580. tunning in range [400 - 700]
    #                 viewpoint_cam.fx = focal
    #                 viewpoint_cam.fy = viewpoint_cam.aspect_ratio * focal
    #                 viewpoint_cam.kappa = 0.0

    #             self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
    #             self.calibration_optimizer.update_focal_learning_rate (lr = 0.1)
    #             self.calib_safe_guard = False

    #             if self.use_gui:
    #                 cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
    #                 self.push_to_gui(cam_cnt)
    #             time.sleep(3)


    #         frozen_states = ( iteration >= self.start_calib_iter and iteration < self.start_calib_iter + 50 )


    #         if iteration == self.start_calib_iter + 50: # change to SGD
    #             lr = self.calibration_optimizer.estimate_step_size()
    #             self.calibration_optimizer = CalibrationOptimizer(self.viewpoint_stack, focal_reference = self.focal_reference, focal_optimizer_type = "SGD")
    #             self.calibration_optimizer.update_focal_learning_rate (lr = lr)
    #             self.calib_safe_guard = True
            

    #         self.gaussians.update_learning_rate(iteration)

    #         # Every 1000 its we increase the levels of SH up to a maximum degree
    #         if iteration % 1000 == 0:
    #             self.gaussians.oneupSHdegree()

    #         # FORWARD
    #         loss = self.compute_loss (
    #                 use_scale_space = frozen_states,
    #                 use_SSIM = (iteration > self.stop_calib_iter) )


    #         iter_end.record()

    #         # calibration step safe-guard
    #         if self.require_calibration and iteration >= self.start_calib_iter and iteration < self.stop_calib_iter:
    #             if self.calib_safe_guard and loss > loss_prev:
    #                 rich.print(f"[bold yellow][Warning]: learning rate is too big! revoke previous step [/bold yellow]")
    #                 # print(f"loss_prev = {loss_prev},   loss = {loss},   current_focal = {self.viewpoint_stack[0].fx}")
    #                 self.calibration_optimizer.undo_focal_step()
    #                 # print(f"\t after revoking, current_fx = {self.viewpoint_stack[0].fx}")

    #                 if (not undo_prev):
    #                     undo_prev = True
    #                     # recompute the loss, as Gaussian/pose update may have changed it
    #                     loss_prev = self.compute_loss (
    #                             use_scale_space = frozen_states,
    #                             use_SSIM = (iteration > self.opt.densify_until_iter) )
    #                 # verify the loss again
    #                 if loss > loss_prev:
    #                     self.calibration_optimizer.update_focal_learning_rate (scale = 0.5)

    #                 with torch.no_grad():
    #                     self.calibration_optimizer.focal_step() # focal step with old gradient
    #                 # print(f"\t update to     , current_fx = {self.viewpoint_stack[0].fx}\n")
    #                 continue


    #         undo_prev = False
    #         if loss < loss_prev:
    #             loss_prev = loss


    #         self.calibration_optimizer.zero_grad() # clear gradient every iteration
    #         self.pose_optimizer.zero_grad() # clear gradient every iteration
    #         self.gaussians.optimizer.zero_grad(set_to_none = True) # clear gradient every iteration

    #         loss.backward()


    #         with torch.no_grad():
    #             # Progress bar
    #             ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    #             if iteration % 10 == 0:
    #                 progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
    #                 progress_bar.update(10)
    #             if iteration == self.opt.iterations:
    #                 progress_bar.close()

    #             # Densification
    #             if True and iteration < self.opt.densify_until_iter:

    #                 self.set_gaussian_densification_stats()

    #                 if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
    #                     sfm_gui.Log("Densify and Prune Gaussians", tag="SFM")
    #                     size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
    #                     self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
                    
    #                 if iteration % self.opt.opacity_reset_interval == 0:
    #                     sfm_gui.Log("Reset opacity of all Gaussians", tag="SFM")
    #                     self.gaussians.reset_opacity()


    #             # calibration step
    #             if self.require_calibration and iteration >= self.start_calib_iter and iteration < self.stop_calib_iter:
    #                 self.calibration_optimizer.focal_step()
    #                 if not frozen_states and self.allow_lens_distortion:
    #                     self.calibration_optimizer.kappa_step()
                
    #             # pose step
    #             if iteration >= self.start_pose_iter and iteration < self.stop_pose_iter:
    #                 self.pose_optimizer.step()
                

    #             # Gaussian step
    #             if iteration >= self.start_gaussian_iter and iteration < self.stop_gaussian_iter:
    #                 self.gaussians.optimizer.step()


    #             if self.use_gui and (iteration % 10 == 0 or frozen_states):
    #                 cam_cnt = (cam_cnt+1) % len(self.viewpoint_stack)
    #                 self.push_to_gui(cam_cnt)
    #                 if frozen_states:
    #                     time.sleep(0.05)



    #     # focal_stack, focal_grad_stack = self.calibration_optimizer.get_focal_statistics(all = True)
    #     # LineDetection(focal_stack[:80], focal_grad_stack[:80]).plot_figure(fname = pathlib.Path.home()/( "focal_cost_function_scale"+str(self.gaussian_scale_t)+".pdf" ) )
         
        




    #     sfm_gui.Log(f"SfM optimization complete with {iteration} iterations.")

    #     if self.use_gui:
    #         self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))  
    #         time.sleep(3.0)




if __name__ == "__main__":

    mp.set_start_method('spawn')
    torch.cuda.set_device(0)


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
    opt.require_calibration = True
    opt.allow_lens_distortion = True


    opt.iterations = 200
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 750
    opt.densify_grad_threshold = 0.0002



    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    cameras_extent = scene.cameras_extent


    N = 10

    viewpoint_stack = scene.getTrainCameras()

    for idx, viewpoint in enumerate(viewpoint_stack):
        viewpoint.uid = idx
    # print_viewpoint_stack(viewpoint_stack)

    ids = [ random.randint(0, len(viewpoint_stack)-1)  for i in range(N) ]
    ids.sort()
    ids = [ 11, 41, 51, 93, 101, 140, 143, 154, 203, 243 ]

    filtered_viewpoint_stack = [ viewpoint_stack[id] for id in ids ]
    viewpoint_stack = filtered_viewpoint_stack


    print_viewpoint_stack(viewpoint_stack)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    

    ## visualization
    use_gui = True
    sfm = SFM(pipe=pipe, use_gui=use_gui, viewpoint_stack=viewpoint_stack, gaussians=gaussians, opt=opt, cameras_extent=cameras_extent)
    sfm.require_calibration = True
    sfm.allow_lens_distortion = True

    sfm.optimize(phase1_iter = 0, phase3_iter = 0, phase2_DBA_iter = 100, phase2_CaliDBA_iter = 10, phase2_CaliDBA_GSS_iter = 0, set_focal_error=0)

    # sfm.optimize(phase1_iter = 0, phase3_iter = 0, phase2_DBA_iter = 0, phase2_CaliDBA_iter = 10, phase2_CaliDBA_GSS_iter = 0, set_focal_error=0)

    (W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images) = sfm.eval_data()

