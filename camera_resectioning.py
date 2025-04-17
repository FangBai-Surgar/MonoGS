
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


from slam_impl.camera_utils import Camera
from slam_impl.pose_utils import update_pose


from gui import gui_utils, sfm_gui
from slam_impl.multiprocessing_utils import FakeQueue, clone_obj


from slam_impl.optimizers import CalibrationOptimizer, PoseOptimizer, LineDetection, lr_exp_decay_helper

from slam_impl.gaussian_scale_space import image_conv_gaussian

from matplotlib import pyplot as plt

import pathlib
import rich
import json
import pickle

from gaussian_viewer import Viewer, create_gaussians_gl


from utils.colmap_utils.gaussian_splatting_utils import assemble_3DGS_cameras_from_3DGS_JSON_file

from utils.matplot_utils import annotate_image, annotate_image_by_table

import itertools
import cv2
import glob
from gaussian_splatting.utils.system_utils import mkdir_p




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



def plot_loss_focal_kappa (focal_stack, kappa_stack, loss_stack, gt_fx = None, gt_kappa = None, fname = "loss_focal_kappa_iters.pdf"):
    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    plt.rcParams["figure.figsize"] = (5, 2.3)
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.edgecolor'] = 'gray'

    color1='mediumblue'
    color2='crimson'
    color3='gray'

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    iterations = np.arange(0, len(focal_stack))

    ax1.plot (iterations, focal_stack, '-',  color=color1, linewidth=0.5)
    ax1r = ax1.twinx()
    ax1r.plot(iterations, kappa_stack, '-',  color=color2, linewidth=0.5)

    ax1.axhline(gt_fx, linestyle="--", color=color1, linewidth=1.0)
    ax1r.axhline(gt_kappa, linestyle="--", color=color2, linewidth=1.0)

    ax2.plot (iterations, loss_stack,  '-',  color=color3, linewidth=0.5)

    if True:

        ax1.set_xlabel(r"iterations", color='k')
        ax1.set_ylabel(r"focal $f_x$", color=color1)
        ax1r.set_ylabel(r"distortion $\kappa$", color=color2)    
        ax1.spines['left'].set_color (color1)
        ax1.spines['right'].set_color (color2)
        # ax1.spines['left'].set_linewidth(2)
        # ax1.spines['right'].set_linewidth(2)
        # ax1.spines['bottom'].set_linewidth(2)
        ax1.tick_params(axis='y', colors=color1)
        ax1r.tick_params(axis='y', colors=color2)

        # ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))        
        # tight axis
        ax1.autoscale(enable=True, axis='x', tight=True)

        ax2.set_xlabel(r"iterations", color='k')
        ax2.set_ylabel(r"loss $L(f_x, \kappa)$", color="k")

        # tight axis
        ax2.autoscale(enable=True, axis='x', tight=True)

    ax1.tick_params(direction='out', length=2, width=2, colors='k', grid_color='r', grid_alpha=0.5)
    ax1r.tick_params(direction='out', length=2, width=2, colors='k', grid_color='r', grid_alpha=0.5)
    ax2.tick_params(direction='out', length=2, width=2, colors='k', grid_color='r', grid_alpha=0.5)

    # tight layout
    # fig.suptitle(" ")
    plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.0)
    plt.savefig(fname=fname)
    plt.close(fig)




def plot_loss_space_focal_length (xdata1, ydata1, yydata1, xdata2, ydata2, yydata2,
                             opts, fname = "focal_cost_function.pdf"):

    gt_datax1 = opts["ground_truth_x1"]
    gt_datax2 = opts["ground_truth_x2"]
    scale_t1  = opts["gaussian_scale_t1"]
    scale_t2  = opts["gaussian_scale_t2"]
    gf_title  = opts["global_title"]

    plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    plt.rcParams["figure.figsize"] = (5, 3)
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.edgecolor'] = 'gray'

    color1='mediumblue'
    color2='chocolate'
    color3='lightgray'

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    ax1.axvline(gt_datax1, linestyle="--",  color=color3)
    ax2.axvline(gt_datax2, linestyle="--", color=color3)
    ax1.axhline(0, linestyle="--", color=color3)
    ax2.axhline(0, linestyle="--", color=color3)

    # Loss
    ax1.scatter(xdata1, ydata1,  s=1, marker='o', color=color1, label=f"scale_t = {scale_t1:.2f}")
    ax1.scatter(xdata2, ydata2,  s=1, marker='o', color=color2, label=f"scale_t = {scale_t2:.2f}")
    # Loss Gradient
    ax2.scatter(xdata1, yydata1, s=1, marker='o', color=color1, label=f"scale_t = {scale_t1:.2f}")
    ax2.scatter(xdata2, yydata2, s=1, marker='o', color=color2, label=f"scale_t = {scale_t2:.2f}")

    if True:

        ax1.set_title(f"loss", fontweight='bold')
        ax1.set_xlabel(r"normalized focal length $f$", color='k')
        ax1.set_ylabel(r"$L(f)$", color="k")

        # tight axis
        ax1.autoscale(enable=True, axis='x', tight=True)
        ax1.autoscale(enable=True, axis='y', tight=False)

        ax2.set_title(f"loss gradient", fontweight='bold')
        ax2.set_xlabel(r"normalized focal length $f$", color='k')
        ax2.set_ylabel(r"$\nabla L(f)$", color="k")

        # tight axis
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax2.autoscale(enable=True, axis='y', tight=False)

    ax1.tick_params(direction='out', length=2, width=2, colors='k', grid_color='r', grid_alpha=0.5)
    ax2.tick_params(direction='out', length=2, width=2, colors='k', grid_color='r', grid_alpha=0.5)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=False, ncol = 2)

    # tight layout
    fig.suptitle(" ")
    plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=0.0)
    plt.savefig(fname=fname)

    plt.show(block=False)
    plt.waitforbuttonpress(10)
    plt.close(fig)




class CameraResectioning(mp.Process):


    def __init__(self, viewpoint_stack = None, gaussians = None, pipe = None, opt = None) -> None:
        self.viewpoint_stack = viewpoint_stack   # list of cameras
        self.gaussians = gaussians   # fixed in camera resectioning
        self.pipe = pipe
        self.opt = opt

        '''
        At initialization, if both kappa and focal are optimzied at the same time, the value of kappa fluctuates.
        THUS, it is better to optimize focal ONLY for some iterations, before JOINTLY optimizing focal and kappa 
        '''
        self.start_kappa_optimization_at_iter = 30  # optimize focal ONLY before this iteration

        self.gaussians.optimizer = None # Do NOT optimize Gaussian

        self.background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        self.rgb_boundary_threshold = 0.01


        self.debug = False

        self.pause = False
        

        self.require_calibration = True
        self.allow_lens_distortion = True

        self.focal_reference = None

        self.calibration_optimizer = None
        self.pose_optimizer = None
        self.calib_safe_guard = False

        self.gaussian_scale_t = None

        self.MODULE_TEST_CALIBRATION = False
        self.add_calib_noise_iter = -1

        self.focal_stack, self.focal_grad_stack, self.kappa_stack, self.kappa_grad_stack, self.loss_stack = [], [], [], [], []



    @staticmethod
    def init_from_3DGS_output_dir(pipe = None, opt = None, base_dir="/hdd/3DGS/train", iter_num=7000):
        """
        read 3DGS rendering output
        """        
        camera_file_path = os.path.join(base_dir, "cameras.json")
        point_cloud_file_path = os.path.join(base_dir, "point_cloud/iteration_"+str(iter_num)+"/point_cloud.ply")

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(point_cloud_file_path)

        viewpoint_stack = assemble_3DGS_cameras_from_3DGS_JSON_file (camera_file_path)

        print(f"Loaded 3DGS data:\n\tnumber of cameras: {len(viewpoint_stack)}\n\tnumber of Gaussians: {len(gaussians.get_xyz)}")

        return  CameraResectioning(viewpoint_stack = viewpoint_stack, gaussians = gaussians, pipe = pipe, opt = opt)


    def set_viewpoint_calibration (self, view_id=0, delta_focal=0.0, delta_kappa=0.0):
        assert ( view_id >= 0 and view_id < len(self.viewpoint_stack) ), f"view_id={view_id} out of range!"
        viewpoint = self.viewpoint_stack[view_id]

        # first set to ground-truth
        viewpoint.fx = viewpoint.fx_init
        viewpoint.fy = viewpoint.fy_init
        viewpoint.kappa = viewpoint.kappa_init
        viewpoint.R = viewpoint.R_gt.clone()
        viewpoint.T = viewpoint.T_gt.clone()

        # new values
        focal = viewpoint.fx + delta_focal
        kappa = viewpoint.kappa + delta_kappa
        
        viewpoint.fx = focal
        viewpoint.fy = focal * viewpoint.aspect_ratio
        viewpoint.kappa = kappa

        # save ground-truth values
        viewpoint.fx_init = viewpoint.fx
        viewpoint.fy_init = viewpoint.fy
        viewpoint.kappa_init = viewpoint.kappa
        viewpoint.R_gt = viewpoint.R.clone()
        viewpoint.T_gt = viewpoint.T.clone()

        # render a distorted image
        render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                            scaling_modifier=1.0,
                            override_color=None,
                            mask=None,)
        image = render_pkg["render"]
        viewpoint.original_image = image.data.clone()
        return viewpoint
    


    def sample_cost_space(self, view_id = 0, num_samples = 100, use_scale_space = False):
        assert ( view_id >= 0 and view_id < len(self.viewpoint_stack) ), f"view_id={view_id} out of range!"
        viewpoint = self.viewpoint_stack[view_id]
        if viewpoint.original_image is None:
            print("viewpoint.original_image is None.")
            return        
        self.debug = True
        
        gt_focal = viewpoint.fx_init
        gt_kappa = viewpoint.kappa_init

        self.focal_stack, self.focal_grad_stack, self.kappa_stack, self.kappa_grad_stack, self.loss_stack = [], [], [], [], []

        _, h, w = viewpoint.original_image.shape
        self.gaussian_scale_t = 0.01 * max(w,h)
        # self.focal_reference = np.sqrt(h*h + w*w)/2
        self.focal_reference = gt_focal


        gt_focal_nml = gt_focal/self.focal_reference
        focal_array = np.linspace(gt_focal_nml - 0.25,  gt_focal_nml + 0.25,   num_samples) * self.focal_reference

        kappa_array = np.linspace(-0.5, 0.5, num_samples)
        kappa_array = [gt_kappa]

        for focal in focal_array:
            for kappa in kappa_array:

                self.focal_stack.append(focal)
                self.kappa_stack.append(kappa)

                viewpoint.fx = focal
                viewpoint.fy = focal * viewpoint.aspect_ratio
                viewpoint.kappa = kappa

                # FORWARD
                loss = self.compute_loss_one_view ( viewpoint, use_scale_space = use_scale_space, use_smooth_l1 = True, use_SSIM = False )
                # BACKWARD
                loss.backward()
                # print(f"loss = {loss.data.cpu().numpy().item()}")
                with torch.no_grad():
                    self.loss_stack.append(loss.data.cpu().numpy().item())
                    focal_grad = viewpoint.cam_focal_delta.grad.cpu().numpy()[0] # * self.focal_reference
                    kappa_grad = viewpoint.cam_kappa_delta.grad.cpu().numpy()[0]
                    self.focal_grad_stack.append(focal_grad)
                    self.kappa_grad_stack.append(kappa_grad)
                    self.zero_calib_grad(viewpoint)

        results = {
            "fx" : viewpoint.fx,
            "fy" : viewpoint.fy,
            "kappa" : viewpoint.kappa,
            "gt_fx" : viewpoint.fx_init,
            "gt_fy" : viewpoint.fy_init,
            "gt_kappa" : viewpoint.kappa_init,
            "R" : viewpoint.R,
            "T" : viewpoint.T,
            "gt_R" : viewpoint.R_gt,
            "gt_T" : viewpoint.T_gt,
            "focal_stack" : np.array(self.focal_stack)/self.focal_reference,
            "focal_grad_stack" : np.array(self.focal_grad_stack)*self.focal_reference,
            "kappa_stack" : np.array(self.kappa_stack),
            "kappa_grad_stack" : np.array(self.kappa_grad_stack),
            "loss_stack" : np.array(self.loss_stack),
            "gaussian_scale_t" : self.gaussian_scale_t if use_scale_space else 0.0,
            "focal_reference" : self.focal_reference
        }
        self.calibration_optimizer = None
        self.pose_optimizer = None
        return results


    def zero_calib_grad(self, viewpoint):
        viewpoint.cam_focal_delta.data.fill_(0)
        viewpoint.cam_kappa_delta.data.fill_(0)
        if viewpoint.cam_focal_delta.grad is not None:
            viewpoint.cam_focal_delta.grad.detach_()
            viewpoint.cam_focal_delta.grad.fill_(0)
        if viewpoint.cam_kappa_delta.grad is not None:
            viewpoint.cam_kappa_delta.grad.detach_()
            viewpoint.cam_kappa_delta.grad.fill_(0)


    def optimize (self, view_id = 0, max_iters = 1000, set_focal_error=None, set_kappa_error=None, update_pose=False, update_calibration = True, scale_space_iters=-1, use_smooth_l1=True):
        assert ( view_id >= 0 and view_id < len(self.viewpoint_stack) ), f"view_id={view_id} out of range!"
        viewpoint = self.viewpoint_stack[view_id]
        if viewpoint.original_image is None:     
            print("viewpoint.original_image is None. Render these images first.")       
            return
        self.debug = False

        _, h, w = viewpoint.original_image.shape
        self.gaussian_scale_t = 0.01 * max(w,h)  # if self.gaussian_scale_t is None else self.gaussian_scale_t
        self.focal_reference = np.sqrt(h*h + w*w)/2 # if self.focal_reference is None else self.focal_reference

        if self.calibration_optimizer is None:            
            self.calibration_optimizer = CalibrationOptimizer([ viewpoint ], focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            self.calibration_optimizer.update_focal_learning_rate (lr = 0.02) # 0.002
            self.calibration_optimizer.update_kappa_learning_rate (lr = 0.01)
            # self.calib_safe_guard = False

        if self.pose_optimizer is None:
            self.pose_optimizer = PoseOptimizer([ viewpoint ])

        """
        use noisy initial value
        """
        if set_focal_error is not None:
            noise_fx = set_focal_error
            focal = viewpoint.fx + noise_fx
            viewpoint.fx = focal
            viewpoint.fy = focal * viewpoint.aspect_ratio
            rich.print(f"[bold red][Notice]: old fx {focal - noise_fx} ====> new fx {focal}.  Noise added {noise_fx}  [/bold red]")

        if set_kappa_error is not None:
            noise_kappa = set_kappa_error
            kappa = viewpoint.kappa + noise_kappa
            viewpoint.kappa = kappa
            rich.print(f"[bold red][Notice]: old kappa {kappa - noise_kappa} ====> new kappa {kappa}.  Noise added {noise_kappa}  [/bold red]")


        sfm_gui.Log("start Camera Resectioning Optimization\n", tag="SFM")        

        self.focal_stack, self.focal_grad_stack, self.kappa_stack, self.kappa_grad_stack, self.loss_stack = [], [], [], [], []
        '''
        Optimization
        '''
        print_viewpoint_stack([ viewpoint ], prefix=f"Camera {view_id}")

        use_scale_space = (scale_space_iters > 0) #initial

        for iteration in range(0, max_iters):
            """
                Disable Gaussian scale space at iter = scale_space_iters
            """
            if (iteration == scale_space_iters):
                use_scale_space = False

            # if (iteration == 100):
            #     lr = self.calibration_optimizer.estimate_step_size()
            #     self.calibration_optimizer = CalibrationOptimizer([ viewpoint ], focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
            #     self.calibration_optimizer.update_focal_learning_rate (lr = 0.02)
            #     self.calibration_optimizer.update_kappa_learning_rate (lr = 0.01)

            if (iteration == 100):
                lr = self.calibration_optimizer.estimate_step_size()
                self.calibration_optimizer = CalibrationOptimizer([ viewpoint ], focal_reference = self.focal_reference, focal_optimizer_type = "Adam")
                self.calibration_optimizer.update_focal_learning_rate (lr = 0.002)
                self.calibration_optimizer.update_kappa_learning_rate (lr = 0.001)
            
            # FORWARD
            loss = self.compute_loss_one_view ( viewpoint, use_scale_space = use_scale_space, use_smooth_l1 = (use_scale_space and use_smooth_l1), use_SSIM = False )            
            # BACKWARD
            loss.backward()
            with torch.no_grad():
                # record iteration info.
                self.loss_stack.append(loss.data.cpu().numpy().item())
                self.focal_stack.append(viewpoint.fx)
                self.kappa_stack.append(viewpoint.kappa)
                focal_grad = viewpoint.cam_focal_delta.grad.cpu().numpy()[0] # * self.focal_reference
                kappa_grad = viewpoint.cam_kappa_delta.grad.cpu().numpy()[0]
                self.focal_grad_stack.append(focal_grad)
                self.kappa_grad_stack.append(kappa_grad)

                # calibration step            
                if update_calibration:
                    # rich.print(f"[bold yellow]After loss.backward: [/bold yellow]{viewpoint.cam_focal_delta.grad=}")
                    self.calibration_optimizer.focal_step()
                    if self.allow_lens_distortion and (iteration >= self.start_kappa_optimization_at_iter):
                        rich.print(f"[bold red]After loss.backward: [/bold red]{viewpoint.cam_kappa_delta.grad=}")
                        self.calibration_optimizer.kappa_step()
                # pose step
                if update_pose:
                    self.pose_optimizer.step()
                self.calibration_optimizer.zero_grad() # clear gradient every iteration
                self.pose_optimizer.zero_grad() # clear gradient every iteration

            print_viewpoint_stack([ viewpoint ], prefix=f"Camera {view_id}")
       

        sfm_gui.Log(f"optimization complete.\n", tag="SFM")
        torch.cuda.synchronize()

        results = {
            "fx" : viewpoint.fx,
            "fy" : viewpoint.fy,
            "kappa" : viewpoint.kappa,
            "gt_fx" : viewpoint.fx_init,
            "gt_fy" : viewpoint.fy_init,
            "gt_kappa" : viewpoint.kappa_init,
            "R" : viewpoint.R.detach().cpu().numpy(),
            "T" : viewpoint.T.detach().cpu().numpy(),
            "gt_R" : viewpoint.R_gt.detach().cpu().numpy(),
            "gt_T" : viewpoint.T_gt.detach().cpu().numpy(),
            "focal_stack" : self.focal_stack,
            "focal_grad_stack" : self.focal_grad_stack,
            "kappa_stack" : self.kappa_stack,
            "kappa_grad_stack" : self.kappa_grad_stack,
            "loss_stack" : self.loss_stack,
            "gaussian_scale_t" : self.gaussian_scale_t if scale_space_iters > 0 else 0.0,
            "focal_reference" : self.focal_reference
        }
        self.calibration_optimizer = None
        self.pose_optimizer = None
        return results


    def show_rendered_images (self, view_id = None, save_to_dir=None, image_name=None, annotate=True,  use_gt_image=False, resize_to_width=None):
        # plt.rcParams["font.family"] = "Arial"
        # plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname':'Times New Roman'}
        for idx, viewpoint in enumerate(self.viewpoint_stack):
            if (view_id is not None) and idx != view_id:
                continue

            if use_gt_image:
                fx, fy, kappa = viewpoint.fx, viewpoint.fy, viewpoint.kappa
                R, T = viewpoint.R.clone(), viewpoint.T.clone()
                viewpoint.fx = viewpoint.fx_init
                viewpoint.fy = viewpoint.fy_init
                viewpoint.kappa = viewpoint.kappa_init
                viewpoint.R = viewpoint.R_gt.clone()
                viewpoint.T = viewpoint.T_gt.clone()
            
            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.background,
                                scaling_modifier=1.0,
                                override_color=None,
                                mask=None,)
            image, viewspace_point_tensor, visibility_filter, radii, opacity, n_touched = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["opacity"], render_pkg["n_touched"]

            if use_gt_image:
                viewpoint.fx = fx
                viewpoint.fy = fy
                viewpoint.kappa = kappa
                viewpoint.R = R
                viewpoint.T = T

            # convert torch tensor to opencv image
            rgb_original = self.tensor2rgb(image)

            # resize the image            
            if resize_to_width is not None:
                h, w, _ = rgb_original.shape
                dim = (resize_to_width, int( resize_to_width*h/w ))
                rgb = cv2.resize(rgb_original, dim, interpolation = cv2.INTER_AREA)
                print(f"resize image [(H, W, C)] from {rgb_original.shape} to {rgb.shape}")
            else:
                rgb = rgb_original

            gt_str, init_str, est_str = "gt", "init", "est"
            
            focal_ground_truth, kappa_ground_truth = viewpoint.fx_init, viewpoint.kappa_init
            focal_estimate, kappa_estimate = viewpoint.fx, viewpoint.kappa

            focal_initial = self.focal_stack[0] if len(self.focal_stack) else viewpoint.fx_init
            kappa_initial = self.kappa_stack[0] if len(self.kappa_stack) else viewpoint.kappa_init

            headers = (
                '*', 'fx', 'k',
                gt_str,   f'{focal_ground_truth:.2f}', f'{kappa_ground_truth:.5f}',
                init_str, f'{focal_initial:.2f}',      f'{kappa_initial:.5f}',
                est_str,  f'{focal_estimate:.2f}',     f'{kappa_estimate:.5f}'
            )
            format_spec = '{:15}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}'
            mytext = format_spec.format(*headers) if annotate else None

            fig, ax, _ = annotate_image_by_table(rgb, cmap=None, mytext = mytext)

            if save_to_dir is not None:
                image_name = "view"+str(idx)+f"_f{viewpoint.fx_init:.2f}_k{viewpoint.kappa_init:.6f}" if image_name is None else image_name
                plt.savefig(os.path.join(save_to_dir, image_name+'.png'), bbox_inches='tight', pad_inches=0)
                plt.close()
                time.sleep(0.01)

                with open( os.path.join(save_to_dir, image_name+'.txt'), "w" ) as myfile:
                    myfile.write(format_spec.format(*headers))


        plt.show(block=False)



    

    """

    subroutines    

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
            beta = 0.1 if self.debug else 0.001
            huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = beta)
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

        return loss
    
 


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
    

    def clean(self):
        self.calibration_optimizer = None
        self.pose_optimizer = None
        self.gaussians = None
        for viewpoint in self.viewpoint_stack:
            viewpoint.clean()
        torch.cuda.empty_cache()
        



def distort_by_opencv (image_file, kappa, fx, fy):
    # https://stackoverflow.com/a/68706787/4926757
    def invert_map(F):
        sh = (F.shape[0], F.shape[1])
        I = np.zeros_like(F)
        I[:,:,1], I[:,:,0] = np.indices(sh)
        P = np.copy(I)
        for i in range(10):
            P += I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        return P

    k_1 = kappa

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)

    h, w, c = img.shape[0], img.shape[1], img.shape[2]


    dist_coeffs = np.array([k_1, 0, 0, 0, 0])  # (k1, k2, p1, p2, k3)
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fx
    camera_matrix[1, 1] = fy
    camera_matrix[0, 2] = (w-1)/2
    camera_matrix[1, 2] = (h-1)/2
    new_camera_matrix = camera_matrix.copy()

    # Compute "Undistort" maps:  
    mapxy, _ = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), m1type=cv2.CV_32FC2)

    # Invert the maps    
    inv_mapxy = invert_map(mapxy)
    inv_mapx = inv_mapxy[:, :, 0]
    inv_mapy = inv_mapxy[:, :, 1]

    # Use the inverted maps
    out_img = cv2.remap(img, inv_mapx, inv_mapy, cv2.INTER_LINEAR)

    cv2.imwrite('view_cv_out_img_' + f"k{kappa:.6f}" + '.png', out_img)

    





if __name__ == "__main__":

    mp.set_start_method('spawn')

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--data_dir", default="/hdd/3DGS/playroom")
    parser.add_argument("--data_iter_num", default=7000) # else 30000

    args = parser.parse_args(sys.argv[1:])

    base_dir = args.data_dir
    iter_num = args.data_iter_num

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)



    

    """
    Test image rendering with given calibratoin parameters
    """
    if False:

        PnP = CameraResectioning.init_from_3DGS_output_dir(pipe = pipe, opt = opt, base_dir=base_dir, iter_num=iter_num)

        view_id = 0

        delta_focal = 0.0
        fx = PnP.viewpoint_stack[view_id].fx
        fy = PnP.viewpoint_stack[view_id].fy
        
        save_to_dir="."

        delta_kappa = 0.0
        PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
        PnP.show_rendered_images(view_id, save_to_dir, annotate=False)
        PnP.set_viewpoint_calibration(view_id, delta_focal=-delta_focal, delta_kappa=-delta_kappa) # cancel previous changes

        image_file = glob.glob( os.path.join(save_to_dir, "view0_*k0.000000.png")  )[0]


        delta_kappa =  0.55
        PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
        PnP.show_rendered_images(view_id, save_to_dir, annotate=False)
        PnP.set_viewpoint_calibration(view_id, delta_focal=-delta_focal, delta_kappa=-delta_kappa) # cancel previous changes
        distort_by_opencv (image_file=image_file, kappa=delta_kappa, fx=fx, fy=fy)


        delta_kappa = -0.5
        PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
        PnP.show_rendered_images(view_id, save_to_dir, annotate=False)
        PnP.set_viewpoint_calibration(view_id, delta_focal=-delta_focal, delta_kappa=-delta_kappa) # cancel previous changes
        distort_by_opencv (image_file=image_file, kappa=delta_kappa, fx=fx, fy=fy)   





    """
    Gaussian Scale Space: Effect on optimization
    """
    if False:

        PnP = CameraResectioning.init_from_3DGS_output_dir(pipe = pipe, opt = opt, base_dir="/hdd/3DGS/train", iter_num=7000)

        view_id = 0
        save_to_dir="."

        delta_focal = 0.0
        delta_kappa = -0.0
        PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)
        PnP.show_rendered_images(view_id, save_to_dir, annotate=False)

        PnP.require_calibration = True
        PnP.allow_lens_distortion = True

        results1 = PnP.sample_cost_space(view_id = 0, num_samples = 200, use_scale_space = False)
        results2 = PnP.sample_cost_space(view_id = 0, num_samples = 200, use_scale_space = True)

        focal_stack1, focal_grad_stack1, gaussian_scale_t1, loss_stack1 = results1["focal_stack"], results1["focal_grad_stack"], results1["gaussian_scale_t"], results1["loss_stack"]
        focal_stack2, focal_grad_stack2, gaussian_scale_t2, loss_stack2 = results2["focal_stack"], results2["focal_grad_stack"], results2["gaussian_scale_t"], results2["loss_stack"]

        opts = {
            "ground_truth_x1" : ( results1["gt_fx"] / results1["focal_reference"] ),
            "ground_truth_x2" : ( results2["gt_fx"] / results2["focal_reference"] ),
            "gaussian_scale_t1" : gaussian_scale_t1,
            "gaussian_scale_t2" : gaussian_scale_t2,
            "global_title" : r"$\nabla L(f) = 2 a f + b $, with Huber loss $\delta =1.0$"
        }
        plot_loss_space_focal_length (focal_stack1, loss_stack1, focal_grad_stack1,
                                 focal_stack2, loss_stack2, focal_grad_stack2,
                                 opts=opts,
                                 fname = "focal_cost_function.pdf")
        
        sys.exit()




    """
    optimization
    Download "Pre-trained Models (14 GB)" from 3DGS repo: https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#training-speed-acceleration

    Execute in command line:

        wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

    """
    view_id_set = np.arange(0, 200, 5).tolist()
    dataset_selected = [ "drjohnson", "playroom", "train", "truck" ]
    # dataset_selected = [  "drjohnson", "truck" ]


    def eval_iteration_convergence (focal_stack, kappa_stack, gt_fx, gt_kappa):
        rle_focal = abs( ( np.array(focal_stack) - gt_fx ) / gt_fx )
        rle_kappa = abs( ( np.array(kappa_stack) - gt_kappa) / gt_kappa )

        success = ( rle_focal[-1] < 0.01 and rle_kappa[-1] < 0.01 )

        for iter in range(len(rle_focal)):
            if rle_focal[iter] < 0.01 and rle_kappa[iter] < 0.01:
                return success, iter
        
        return success, len(rle_focal)


    if True:

        max_iters = 500
        dataset_root_dir = "/hdd/3DGS"

        if False:
            for dataset_name in [ "drjohnson", "playroom", "train", "truck",  "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill", "bicycle"  ]:
                base_dir = os.path.join(dataset_root_dir, dataset_name)
                save_to_dir = os.path.join("result_pnp", dataset_name)
                mkdir_p(save_to_dir)
                PnP = None
                torch.cuda.empty_cache()
                PnP = CameraResectioning.init_from_3DGS_output_dir(pipe = pipe, opt = opt, base_dir=base_dir, iter_num=iter_num)
                PnP.set_viewpoint_calibration(view_id=0, delta_focal=0.0, delta_kappa=0.0)
                PnP.show_rendered_images(view_id=0, save_to_dir=save_to_dir, annotate=False, use_gt_image=True, resize_to_width=640)

        try:
            with open(os.path.join( "result_pnp", 'results_dict.pkl'), 'rb') as fp:
                results_dict = pickle.load(fp)
        except:
            results_dict = {}


        for dataset_name in dataset_selected:

            base_dir = os.path.join(dataset_root_dir, dataset_name)

            results_dict[dataset_name] = {}
            """
            different optimization strategies:
            """
            for scale_space_iters in [-1, 100]:
                for use_smooth_l1 in [True, False]:

                    gss_str = 'Y' if (scale_space_iters > 0) else 'N'
                    sl1_str = 'Y' if use_smooth_l1 else 'N'
                    gss_sl1_str = "gss" + gss_str + "_sl1" + sl1_str

                    results_dict[dataset_name][gss_sl1_str] = {}
                    """
                    different calibration parameters:
                    """
                    for delta_focal_ratio in [-0.333, 1.0]:
                        for delta_kappa in [-0.3, 0.3]:

                            focal_str = "U" if (delta_focal_ratio>0) else "D"
                            kappa_str = "U" if (delta_kappa>0) else "D"
                            focal_kappa_str = "f" + focal_str + "_k" + kappa_str

                            results_dict[dataset_name][gss_sl1_str][focal_kappa_str] = {}
                            '''
                            views
                            '''
                            for view_id in view_id_set:

                                image_name = "view" + str(view_id) + "_" + focal_kappa_str

                                PnP = None
                                torch.cuda.empty_cache()

                                save_to_dir = os.path.join("result_pnp", dataset_name, gss_sl1_str)
                                mkdir_p(save_to_dir)

                                PnP = CameraResectioning.init_from_3DGS_output_dir(pipe = pipe, opt = opt, base_dir=base_dir, iter_num=iter_num)

                                focal_ref = PnP.viewpoint_stack[view_id].fx
                                delta_focal = delta_focal_ratio*focal_ref

                                PnP.set_viewpoint_calibration(view_id, delta_focal=delta_focal, delta_kappa=delta_kappa)

                                results = PnP.optimize (view_id, max_iters = max_iters, set_focal_error=-delta_focal, set_kappa_error=-delta_kappa,
                                            update_pose=False, update_calibration=True, scale_space_iters=scale_space_iters, use_smooth_l1=use_smooth_l1)
                        
                                PnP.show_rendered_images(view_id, save_to_dir, image_name=image_name, annotate=False, use_gt_image=True, resize_to_width=640)

                                """
                                Evaluation. put it in a function taking results as the argument?
                                """
                                fx_est, fy_est, kappa_est = results["fx"], results["fy"], results["kappa"]
                                fx_gt, fy_gt, kappa_gt = results["gt_fx"], results["gt_fy"], results["gt_kappa"]
                                aspect_ratio = fy_gt / fx_gt
                                fx_init, fy_init, kappa_init = results["focal_stack"][0], results["focal_stack"][0]*aspect_ratio, results["kappa_stack"][0]

                                # relative error. err_fx = err_fy, since fy = aspect_ratio*fx
                                err_fx = (fx_est - fx_gt) / fx_gt
                                err_fy = (fy_est - fy_gt) / fy_gt
                                err_kappa = (kappa_est - kappa_gt) / kappa_gt

                                results["error"] = [err_fx, err_kappa]  # { "fx" : err_fx,   "kappa" : err_kappa }
                                results["succeed"] = 1 if ( abs(err_fx) < 0.01 and abs(err_kappa) < 0.01 ) else 0

                                results_dict[dataset_name][gss_sl1_str][focal_kappa_str][view_id] = results

                                '''
                                print result
                                '''
                                headers = (
                                    '*', 'fx', 'fy', 'k', 
                                    "gt",   f'{fx_gt:.2f}',   f'{fy_gt:.2f}',   f'{kappa_gt:.5f}',
                                    "init", f'{fx_init:.2f}', f'{fy_init:.2f}', f'{kappa_init:.5f}',
                                    "est",  f'{fx_est:.2f}',  f'{fy_est:.2f}',  f'{kappa_est:.5f}',
                                    "error(f)",f'{err_fx:.5f}',  f'{err_fy:.5f}',  f'{err_kappa:.5f}',
                                    "error(%)",f'{err_fx:.3%}',  f'{err_fy:.3%}',  f'{err_kappa:.3%}'
                                )
                                format_spec = '{:15}  {:>10}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}  {:>10}\n{:15}  {:>10}  {:>10}  {:>10}'
                                print(format_spec.format(*headers))
                                with open( os.path.join(save_to_dir, image_name+'.txt'), "w" ) as myfile:
                                    myfile.write(format_spec.format(*headers))

                                '''
                                plot figure
                                '''
                                plot_loss_focal_kappa ( focal_stack=results["focal_stack"].copy()[:250],
                                                        kappa_stack=results["kappa_stack"].copy()[:250],
                                                        loss_stack=results["loss_stack"].copy()[:250],
                                                        gt_fx=fx_gt, gt_kappa=kappa_gt,
                                                        fname = os.path.join(save_to_dir, image_name+'_it250.pdf')
                                                       )
                                plot_loss_focal_kappa ( focal_stack=results["focal_stack"].copy()[:500],
                                                        kappa_stack=results["kappa_stack"].copy()[:500],
                                                        loss_stack=results["loss_stack"].copy()[:500],
                                                        gt_fx=fx_gt, gt_kappa=kappa_gt,
                                                        fname = os.path.join(save_to_dir, image_name+'_it500.pdf')
                                                       )
                                plot_loss_focal_kappa ( focal_stack=results["focal_stack"].copy(),
                                                        kappa_stack=results["kappa_stack"].copy(),
                                                        loss_stack=results["loss_stack"].copy(),
                                                        gt_fx=fx_gt, gt_kappa=kappa_gt,
                                                        fname = os.path.join(save_to_dir, image_name+'.pdf')
                                                       )

                                with open(os.path.join( "result_pnp", 'results_dict.pkl'), 'wb') as fp:
                                    pickle.dump(results_dict, fp)
                                    print('dictionary saved successfully to file')

                                PnP.clean()


    else:        

        with open(os.path.join( "result_pnp", 'results_dict.pkl'), 'rb') as fp:
            results_dict = pickle.load(fp)

        """
        process result to tables
        """
        gss_str_Y, gss_str_N = "gssY_sl1N", "gssN_sl1N"

        fU_kU_dict, fU_kD_dict, fD_kU_dict, fD_kD_dict = {}, {}, {}, {}
        for view_id in view_id_set:
            fU_kU, fU_kD, fD_kU, fD_kD = [], [], [], []
            for dataset_name in dataset_selected:
                data = results_dict[dataset_name]
                # [fx, kappa] relative error
                fU_kU.append( [ data[gss_str_Y]['fU_kU'][view_id]["error"],  data[gss_str_N]['fU_kU'][view_id]["error"] ] )
                fU_kD.append( [ data[gss_str_Y]['fU_kD'][view_id]["error"],  data[gss_str_N]['fU_kD'][view_id]["error"] ] )
                fD_kU.append( [ data[gss_str_Y]['fD_kU'][view_id]["error"],  data[gss_str_N]['fD_kU'][view_id]["error"] ] )
                fD_kD.append( [ data[gss_str_Y]['fD_kD'][view_id]["error"],  data[gss_str_N]['fD_kD'][view_id]["error"] ] )
            fU_kU_dict[view_id] = fU_kU
            fU_kD_dict[view_id] = fU_kD
            fD_kU_dict[view_id] = fD_kU
            fD_kD_dict[view_id] = fD_kD


        fU_kU_sr, fU_kD_sr, fD_kU_sr, fD_kD_sr = [], [], [], []
        fU_kU_it, fU_kD_it, fD_kU_it, fD_kD_it = [], [], [], []
        for dataset_name in dataset_selected:
            data = results_dict[dataset_name]

            calib_fk_str = 'fU_kU'
            sr_gssY, sr_gssN = 0, 0
            it_gssY, it_gssN = 0, 0
            for view_id in view_id_set:
                # with GSS
                result = data[gss_str_Y][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssY += 1               if success else 0
                it_gssY += converge_iters  if success else 0
                # with/o GSS
                result = data[gss_str_N][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssN += 1               if success else 0
                it_gssN += converge_iters  if success else 0
            # it_gssY, it_gssN = it_gssY, it_gssN
            fU_kU_it.append( [ it_gssY/sr_gssY, it_gssN/sr_gssN ] )
            fU_kU_sr.append( [ sr_gssY/len(view_id_set), sr_gssN/len(view_id_set) ] )


            calib_fk_str = 'fU_kD'
            sr_gssY, sr_gssN = 0, 0
            it_gssY, it_gssN = 0, 0
            for view_id in view_id_set:
                # with GSS
                result = data[gss_str_Y][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssY += 1               if success else 0
                it_gssY += converge_iters  if success else 0
                # with/o GSS
                result = data[gss_str_N][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssN += 1               if success else 0
                it_gssN += converge_iters  if success else 0
            fU_kD_it.append( [ it_gssY/sr_gssY, it_gssN/sr_gssN ] )
            fU_kD_sr.append( [ sr_gssY/len(view_id_set), sr_gssN/len(view_id_set) ] )


            calib_fk_str = 'fD_kU'
            sr_gssY, sr_gssN = 0, 0
            it_gssY, it_gssN = 0, 0
            for view_id in view_id_set:
                # with GSS
                result = data[gss_str_Y][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssY += 1               if success else 0
                it_gssY += converge_iters  if success else 0
                # with/o GSS
                result = data[gss_str_N][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssN += 1               if success else 0
                it_gssN += converge_iters  if success else 0
            fD_kU_it.append( [ it_gssY/sr_gssY, it_gssN/sr_gssN ] )
            fD_kU_sr.append( [ sr_gssY/len(view_id_set), sr_gssN/len(view_id_set) ] )


            calib_fk_str = 'fD_kD'
            sr_gssY, sr_gssN = 0, 0
            it_gssY, it_gssN = 0, 0
            for view_id in view_id_set:
                # with GSS
                result = data[gss_str_Y][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssY += 1               if success else 0
                it_gssY += converge_iters  if success else 0
                # with/o GSS
                result = data[gss_str_N][calib_fk_str][view_id]
                (success, converge_iters) = eval_iteration_convergence (focal_stack=result["focal_stack"], kappa_stack=result["kappa_stack"], gt_fx=result["gt_fx"], gt_kappa=result["gt_kappa"])
                sr_gssN += 1               if success else 0
                it_gssN += converge_iters  if success else 0
            fD_kD_it.append( [ it_gssY/sr_gssY, it_gssN/sr_gssN ] )
            fD_kD_sr.append( [ sr_gssY/len(view_id_set), sr_gssN/len(view_id_set) ] )



        fU_kU_str = "$f_x \\uparrow$ $\\kappa \\uparrow $"
        fU_kD_str = "$f_x \\uparrow$ $\\kappa \\downarrow$"
        fD_kU_str = "$f_x \\downarrow$ $\\kappa \\uparrow$"
        fD_kD_str = "$f_x \\downarrow$ $\\kappa \\downarrow$"
        print_prefix_str =        [ fU_kU_str,     fU_kD_str,     fD_kU_str,     fD_kD_str    ]

      
        for view_id in view_id_set:
            fU_kU = fU_kU_dict[view_id]
            fU_kD = fU_kD_dict[view_id]
            fD_kU = fD_kU_dict[view_id]
            fD_kD = fD_kD_dict[view_id]
            
            rich.print(f"\n{dataset_selected=}")
            rich.print(f"{gss_str_Y=} ([bold red]fx / kappa[/bold red])  [bold red]&[/bold red]  {gss_str_N=} ([bold red]fx / kappa[/bold red])")
            rich.print(f"{view_id=}")

            for id, vals in enumerate([ fU_kU,  fU_kD,  fD_kU,  fD_kD ]):
                vals_v = list( itertools.chain.from_iterable(vals) )              
                pref = print_prefix_str[id]
                rich.print( pref, " & ", "  &  ".join( f"{x[0]:.2f}\\permil / {x[1]:.2f}\\permil" for x in np.array(vals_v) * 1000  ),  " \\\\" )



        rich.print(f"\n{dataset_selected=}")
        rich.print(f"{gss_str_Y=} ([bold red]success-rate[/bold red])  [bold red]&[/bold red]  {gss_str_N=} ([bold red]success-rate[/bold red])")

        for pref, vals, in zip(print_prefix_str, [ fU_kU_sr,  fU_kD_sr,  fD_kU_sr,  fD_kD_sr ]):
            vals_v = list( itertools.chain.from_iterable(vals) )
            rich.print( pref, " & ", "  &  ".join( f"{x:.3f}" for x in vals_v  ),  " \\\\" )


        rich.print(f"\n{dataset_selected=}")
        rich.print(f"{gss_str_Y=} ([bold red]iterations[/bold red])  [bold red]&[/bold red]  {gss_str_N=} ([bold red]iterations[/bold red])")
        for pref, vals, in zip(print_prefix_str, [ fU_kU_it,  fU_kD_it,  fD_kU_it,  fD_kD_it ]):
            vals_v = list( itertools.chain.from_iterable(vals) )
            rich.print( pref, " & ", "  &  ".join( f"{x:.0f}" for x in vals_v  ),  " \\\\" )


        rich.print(f"\n{dataset_selected=}")
        rich.print(f"{gss_str_Y=} ([bold red]success-rate/iterations[/bold red])  [bold red]&[/bold red]  {gss_str_N=} ([bold red]success-rate/iterations[/bold red])")
        for pref, vals_sr, vals_it in zip(print_prefix_str, [ fU_kU_sr,  fU_kD_sr,  fD_kU_sr,  fD_kD_sr ],  [ fU_kU_it,  fU_kD_it,  fD_kU_it,  fD_kD_it ]):
            vals_v_sr = list( itertools.chain.from_iterable(vals_sr) )
            vals_v_it = list( itertools.chain.from_iterable(vals_it) )
            rich.print( pref, " & ", "  &  ".join( f"{sr*100:.1f}\\% / {it:.0f}" for sr, it in zip(vals_v_sr, vals_v_it)  ),  " \\\\" )


