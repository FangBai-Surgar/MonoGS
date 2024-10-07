
import torch
import torch.optim.lr_scheduler as lr_scheduler

from utils.pose_utils import update_pose

import numpy as np

from numpy.polynomial import Polynomial, Chebyshev

import matplotlib.pyplot as plt

import pathlib

import rich
from rich.console import Console





class CalibrationOptimizer:


    def __init__(self, viewpoint_stack, focal_reference = None, focal_optimizer_type = "Adam") -> None:

        self.viewpoint_stack = viewpoint_stack

        self.calibration_groups = {}
        self.focal_delta_groups = {}
        self.kappa_delta_groups = {}

        self.focal_optimizer = None
        self.kappa_optimizer = None

        self.current_calib_id = -1

        self.focal_optimizer_type = focal_optimizer_type

        self.__init_calibration_groups()
        self.__init_current_calibration_id()
        self.__init_optimizers()

        self.focal_grad_stack = []
        self.focal_stack = []

        self.num_line_elements = 20

        self.update_gaussian_scale_t = False

        self.FOCAL_LENGTH_RANGE = [0, 2000]

        self.focal_gradient_normalizer = focal_reference if focal_reference is not None else viewpoint_stack[0].fx
        
        
    def __init_calibration_groups(self):
        self.calibration_groups = {}
        for viewpoint_cam in self.viewpoint_stack:
            calib_id = viewpoint_cam.calibration_identifier
            if calib_id not in self.calibration_groups:
                self.calibration_groups[ calib_id ] = []
            self.calibration_groups[ calib_id ].append(viewpoint_cam)
        for calib_id, cam_stack in self.calibration_groups.items():
            # gradients to these variables will be computed manually, thus requires_grad = False
            self.focal_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=False, device=cam_stack[0].device)
            self.kappa_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=False, device=cam_stack[0].device)


    def __init_current_calibration_id(self, current_calib_id = None):
        if current_calib_id is not None:
            self.current_calib_id = current_calib_id
        else:
            self.current_calib_id = -1
            for calib_id, cam_stack in self.calibration_groups.items():
                num_views = len(cam_stack)
                if calib_id > self.current_calib_id and num_views >= 1:
                    self.current_calib_id = calib_id


    def __init_optimizers(self):
        focal_opt_params = []
        kappa_opt_params = []
        for calib_id, cam_stack in self.calibration_groups.items():
            focal_opt_params.append(
                    {
                        "params": [ self.focal_delta_groups [ calib_id ] ],
                        "lr": 0.01,
                        "name": "calibration_f_{}".format(calib_id),
                    }
                )
            kappa_opt_params.append(
                    {
                        "params": [ self.kappa_delta_groups [ calib_id ] ],
                        "lr": 0.0001,
                        "name": "calibration_k_{}".format(calib_id),
                    }
                )
        if self.focal_optimizer_type  == "SGD" or self.focal_optimizer_type  == "sgd":
            self.focal_optimizer = torch.optim.SGD(focal_opt_params)
        elif self.focal_optimizer_type  == "Adam" or self.focal_optimizer_type  == "adam" or self.focal_optimizer_type  == "ADAM":
            self.focal_optimizer = torch.optim.Adam(focal_opt_params)
        else:
            raise TypeError("Only SGD and Adam are supported for focal length optimization!")
        
        self.kappa_optimizer = torch.optim.Adam(kappa_opt_params)



    # put it under .grad? to be used with optimizers
    def __update_focal_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():
            self.focal_delta_groups [ calib_id ].data.fill_(0)
            if self.focal_delta_groups [ calib_id ].grad is None:
                self.focal_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)
            else:
                self.focal_delta_groups [ calib_id ].grad.fill_(0)

            for viewpoint_cam in cam_stack:
                self.focal_delta_groups [ calib_id ].grad += viewpoint_cam.cam_focal_delta.grad
            
            # focal_gradient_normalizer: is a guessed working focal length
            # also normalize the gradient as per camera, to help with finding stable tuning parameters, as updates is implemented per camera
            self.focal_delta_groups [ calib_id ].grad *= self.focal_gradient_normalizer #/ len(cam_stack)  # normalized_focal_grad = real_focal_grad * normalizer 



    # put it under .grad? to be used with optimizers
    def __update_kappa_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():
            self.kappa_delta_groups [ calib_id ].data.fill_(0)
            if self.kappa_delta_groups [ calib_id ].grad is None:
                self.kappa_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)
            else:
                self.kappa_delta_groups [ calib_id ].grad.fill_(0)

            for viewpoint_cam in cam_stack:
                self.kappa_delta_groups [ calib_id ].grad += viewpoint_cam.cam_kappa_delta.grad
            # also normalize the gradient as per camera, to help with finding stable tuning parameters, as updates is implemented per camera
            # self.kappa_delta_groups [ calib_id ].grad *=  ( 1.0 / len(cam_stack) )



    # update cameras with calibration_identifier
    def __update_focal_estimates (self, calibration_identifier):
        for calib_id, cam_stack in self.calibration_groups.items():
            if calib_id == calibration_identifier:
                focal_delta_normalized = self.focal_delta_groups [ calib_id ].data.cpu().numpy()[0]
                focal_delta = focal_delta_normalized * self.focal_gradient_normalizer  # real_focal = normalized_focal * normalizer
                focal_grad_normalized  = self.focal_delta_groups [ calib_id ].grad.cpu().numpy()[0]
                for viewpoint_cam in cam_stack:
                    focal = viewpoint_cam.fx
                    viewpoint_cam.fx += focal_delta
                    viewpoint_cam.fy += viewpoint_cam.aspect_ratio * focal_delta
                print(f">> opt_focal: {viewpoint_cam.fx:.3f}, df: {focal_delta:.4f}, df_normalized: {focal_delta_normalized:.7f}, grad_normalized: {focal_grad_normalized:.7f}")
                return focal/self.focal_gradient_normalizer, focal_grad_normalized



    # update cameras' with calibration_identifier
    def __update_kappa_estimates (self, calibration_identifier):
        for calib_id, cam_stack in self.calibration_groups.items():
            if calib_id == calibration_identifier:
                kappa_delta = self.kappa_delta_groups [ calib_id ].data.cpu().numpy()[0]
                kappa_grad  = self.kappa_delta_groups [ calib_id ].grad.cpu().numpy()[0]
                # print(f">>opt_kappa={kappa:.6f}, update={kappa_delta:.6f}, gradient={kappa_grad:.7f}")
                for viewpoint_cam in cam_stack:
                    viewpoint_cam.kappa += kappa_delta
                return kappa_grad

    

    ## estimate Lipschtz constant and maximum stepsize
    def estimate_step_size (self):
        step_size = None
        if type(self.focal_optimizer).__name__  == 'SGD':
            calib_id = self.current_calib_id
            for param_group in self.focal_optimizer.param_groups:
                if param_group["name"] == "calibration_f_{}".format(calib_id):
                    step_size = param_group["lr"]
        else:
            focal_stack, focal_grad_stack = self.get_focal_statistics()
            if focal_stack is None or len(focal_stack) == 0:
                return None        
            L = LineDetection(focal_stack, focal_grad_stack).estimate_Lipschitz_constant()
            step_size = 1.0 / L
        return step_size


    def focal_step(self):
        self.__update_focal_gradients()        
        self.focal_optimizer.step()
        (focal, focal_grad) = self.__update_focal_estimates (calibration_identifier = self.current_calib_id)

        if self.num_line_elements > 0:
            self.focal_grad_stack.append(focal_grad)
            self.focal_stack.append(focal)
        
        converged = ( np.abs(focal_grad) < 0.00001)
        return converged
    

    def undo_focal_step(self):

        self.focal_grad_stack.pop()
        prev_focal_normalized = self.focal_stack.pop()
        prev_focal = prev_focal_normalized * self.focal_gradient_normalizer

        calib_id = self.current_calib_id
        cam_stack = self.calibration_groups[ calib_id ]
        for viewpoint_cam in cam_stack:
            viewpoint_cam.fx = prev_focal
            viewpoint_cam.fy = viewpoint_cam.aspect_ratio * prev_focal



    def kappa_step(self):
        self.__update_kappa_gradients()
        self.kappa_optimizer.step()
        self.__update_kappa_estimates (calibration_identifier = self.current_calib_id)



    def zero_grad(self, set_to_none=True):
        self.focal_optimizer.zero_grad(set_to_none=set_to_none)
        self.kappa_optimizer.zero_grad(set_to_none=set_to_none)
        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)
            if viewpoint_cam.cam_focal_delta.grad is not None:
                # viewpoint_cam.cam_focal_delta.grad.detach_()
                viewpoint_cam.cam_focal_delta.grad.fill_(0)
            if viewpoint_cam.cam_kappa_delta.grad is not None:
                # viewpoint_cam.cam_kappa_delta.grad.detach_()
                viewpoint_cam.cam_kappa_delta.grad.fill_(0)



    def update_focal_learning_rate (self, lr = None, scale = None):
        for param_group in self.focal_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr >= 0.00001 else lr
            rich.print("[bold green]focal_optimizer: update learning rate to:[/bold green]", param_group["lr"])



    def update_kappa_learning_rate (self, lr = None, scale = None):
        for param_group in self.kappa_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr >= 0.00001 else lr
            rich.print("[bold green]kappa_optimizer: update learning rate to:[/bold green]", param_group["lr"])



    def get_focal_statistics (self, all = False):
        if all:
            focal_grad_stack  = np.array(self.focal_grad_stack)
            focal_stack = np.array(self.focal_stack)
        else:
            focal_grad_stack  = np.array(self.focal_grad_stack[-self.num_line_elements:] )
            focal_stack = np.array(self.focal_stack[-self.num_line_elements:])
        return focal_stack, focal_grad_stack






class PoseOptimizer:

    def __init__(self, viewpoint_stack) -> None:

        self.viewpoint_stack = viewpoint_stack

        self.pose_optimizer = None

        self.__init_optimizer()
        self.zero_grad()



    def __init_optimizer(self):
        pose_opt_params = []
        for viewpoint_cam in self.viewpoint_stack:
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_rot_delta],
                    "lr": 0.003,
                    "name": "rot_{}".format(viewpoint_cam.uid),
                }
            )
            pose_opt_params.append(
                {
                    "params": [viewpoint_cam.cam_trans_delta],
                    "lr": 0.001,
                    "name": "trans_{}".format(viewpoint_cam.uid),
                }
            )
        self.pose_optimizer = torch.optim.Adam(pose_opt_params)
        self.pose_optimizer.zero_grad()



    def step(self):
        self.pose_optimizer.step()
        for viewpoint_cam in self.viewpoint_stack:
            if viewpoint_cam.uid != 0:
                update_pose(viewpoint_cam)



    def zero_grad(self, set_to_none=True):
        self.pose_optimizer.zero_grad(set_to_none=set_to_none)
        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.cam_rot_delta.data.fill_(0)
            viewpoint_cam.cam_trans_delta.data.fill_(0)






class LineDetection:

    def __init__(self, xdata, ydata, deg = 5) -> None:

        self.xdata = xdata
        self.ydata = ydata
        
        self.poly = Chebyshev.fit(xdata, ydata, deg=deg)
        self.poly_deriv = self.poly.deriv(1)

        self.ygrad = self.poly_deriv(xdata) # 2*a per point estimate
        self.hessian = self.ygrad[ - len(self.ygrad) // 5 ] # 2*a global estimate. chose a value close to the end
    

    def compute_newton_update (self, grad=None):        
        if grad is not None:
            newton_update = - grad / self.hessian
            return newton_update
        else:
            newton_update = - self.ydata / self.hessian
            newton_est_opt = self.xdata + newton_update
            return newton_update, newton_est_opt
        

    # if g(x) = x^T A x + b x = c, then L = 2 \sigma_max (A)
    # so, g(x) = ax^2 + bx + c, then L = 2a
    def estimate_Lipschitz_constant (self):
        return np.abs(self.hessian)


    def plot_figure (self, fname = pathlib.Path.home()/"focal_cost_function.pdf"):

        newton_update, newton_est_opt = self.compute_newton_update()

        print(f"xdata = {self.xdata}\n")
        print(f"ydata = {self.ydata}\n")
        print(f"ygrad = 2a = {self.ygrad}\n")
        print(f"newton_update = {newton_update}\n")
        print(f"newton_estimate_optimal = {newton_est_opt}\n")


        plt.rcParams['text.usetex'] = True
        plt.rcParams["figure.figsize"] = (8,3)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        color1 = 'r'
        color2 = 'b'

        ax1r = ax1.twinx()
        ax1.plot(self.xdata, self.ydata, '+', color=color1)
        # ax1.scatter(self.xdata[:50], self.ydata[:50])
        xx, yy = self.poly.linspace()
        ax1.plot(xx, yy, lw=2, color=color1)
        ax1r.plot(self.xdata, self.ygrad, '*', color=color2)
        xxd, yyd = self.poly_deriv.linspace()
        ax1r.plot(xxd, yyd, lw=2, color=color2)
        ax1.set_title(r"$\nabla L(f) = 2 a f + b $")
        ax1.set_xlabel(r"current focal length $f$", color='k')
        ax1.set_ylabel(r"$\nabla L(f)$", color=color1)
        ax1r.set_ylabel(r"$\nabla^2 L(f) = 2a$", color=color2)
        ax1.spines['left'].set_color (color1)
        ax1.spines['right'].set_color (color2)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['right'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.tick_params(axis='y', colors=color1)
        ax1r.tick_params(axis='y', colors=color2)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # tight axis
        ax1.autoscale(enable=True, axis='x', tight=False)
        ax1.autoscale(enable=True, axis='y', tight=False)



        color1 = 'r'
        color2 = 'b'

        ax2r = ax2.twinx()
        ax2.plot(self.xdata, newton_est_opt, lw=2, color=color1)
        ax2r.plot(self.xdata, newton_update, '*', color=color2)
        ax2.set_title(r"$f^{\star} = f - \nabla L(f) / (2a) $")
        ax2.set_xlabel(r"current focal length $f$", color='k')
        ax2.set_ylabel(r"Newton estimate", color=color1)
        ax2r.set_ylabel(r"Newton update", color=color2)
        ax2.spines['left'].set_color (color1)
        ax2.spines['right'].set_color (color2)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['right'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.tick_params(axis='y', colors=color1)
        ax2r.tick_params(axis='y', colors=color2)
        # tight axis
        ax2.autoscale(enable=True, axis='x', tight=False)
        ax2.autoscale(enable=True, axis='y', tight=False)


        # tight layout
        plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=0.0)

        plt.savefig(fname=fname)

        plt.show(block=False)
        plt.waitforbuttonpress(10)
        plt.close(fig)






def lr_exp_decay_helper(
    step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels (version inside Guassian splatting)

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


