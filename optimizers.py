
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


    def __init__(self, viewpoint_stack) -> None:

        self.viewpoint_stack = viewpoint_stack

        self.calibration_groups = {}
        self.focal_delta_groups = {}
        self.kappa_delta_groups = {}

        self.focal_optimizer = None
        self.kappa_optimizer = None

        self.__init_calibration_groups()
        self.__init_optimizers()
        self.zero_grad()


        self.focal_grad_stack = []
        self.focal_stack = []

        self.num_line_elements = 20
        self.maximum_newton_steps = 2

        self.update_gaussian_scale_t = False

        self.FOCAL_LENGTH_RANGE = [0, 2000]


    def __init_calibration_groups(self):
        self.calibration_groups = {}
        for viewpoint_cam in self.viewpoint_stack:
            calib_id = viewpoint_cam.calibration_identifier
            if calib_id not in self.calibration_groups:
                self.calibration_groups[ calib_id ] = []
            self.calibration_groups[ calib_id ].append(viewpoint_cam)
        for calib_id, cam_stack in self.calibration_groups.items():
            self.focal_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=True, device=cam_stack[0].device)
            self.kappa_delta_groups [ calib_id ] = torch.tensor([0.0], requires_grad=True, device=cam_stack[0].device)
            self.focal_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)
            self.kappa_delta_groups [ calib_id ].grad = torch.tensor([0.0], device=cam_stack[0].device)



    def __init_optimizers(self):
        focal_opt_params = []
        kappa_opt_params = []
        for calib_id, cam_stack in self.calibration_groups.items():
            focal_opt_params.append(
                    {
                        "params": [ self.focal_delta_groups [ calib_id ] ],
                        "lr": 0.1,
                        "name": "calibration_f_{}".format(calib_id),
                    }
                )
            kappa_opt_params.append(
                    {
                        "params": [ self.kappa_delta_groups [ calib_id ] ],
                        "lr": 0.001,
                        "name": "calibration_k_{}".format(calib_id),
                    }
                )
        self.focal_optimizer = torch.optim.NAdam(focal_opt_params)
        self.kappa_optimizer = torch.optim.NAdam(kappa_opt_params)
        
        



    # put it under .grad? to be used with optimizers
    def __update_focal_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():

            self.focal_delta_groups [ calib_id ].data.fill_(0)
            self.focal_delta_groups [ calib_id ].grad.zero_()

            for viewpoint_cam in cam_stack:
                self.focal_delta_groups [ calib_id ].grad += viewpoint_cam.cam_focal_delta.grad



    # put it under .grad? to be used with optimizers
    def __update_kappa_gradients (self):
        for calib_id, cam_stack in self.calibration_groups.items():

            self.kappa_delta_groups [ calib_id ].data.fill_(0)
            self.kappa_delta_groups [ calib_id ].grad.zero_()

            for viewpoint_cam in cam_stack:
                self.kappa_delta_groups [ calib_id ].grad += viewpoint_cam.cam_kappa_delta.grad



    def focal_step(self, loss=None):
        self.__update_focal_gradients()

        focal_grad_vec = []
        focal_vec = []

        # L-BFGS closure
        def closure():
            return loss

        converged = False

        newton_update_error = False
        # implement a Newton step by estimating Hessian from line fitting of History data (focals, focal_grads)
        if self.maximum_newton_steps > 0 and self.num_line_elements > 0 and len(self.focal_stack) and len(self.focal_stack) % self.num_line_elements == 0:
            focal_stack, focal_grad_stack = self.get_focal_statistics()
            for focals, focal_grads, (calib_id, cam_stack) in zip(focal_stack, focal_grad_stack, self.calibration_groups.items()):                                
                focal_grad  = self.focal_delta_groups [ calib_id ].grad.cpu().numpy()[0]
                newton_update = LineDetection(focals, focal_grads).compute_newton_update(grad = focal_grad)
                test_focal = cam_stack[0].fx + newton_update
                if (test_focal < self.FOCAL_LENGTH_RANGE[0] or test_focal > self.FOCAL_LENGTH_RANGE[1]):
                    newton_update_error = True
                    self.focal_optimizer.step()                    
                else:
                    self.focal_delta_groups [ calib_id ].data.fill_(newton_update)
            if not newton_update_error:
                rich.print(f"\n[bold magenta]Newton update step[/bold magenta]")
                self.maximum_newton_steps -= 1
                self.update_gaussian_scale_t = True
                # decrease learning rate after Newton steps
                # if self.maximum_newton_steps == 0:
                #     self.update_focal_learning_rate(lr = None, scale = 0.1)

        else:

            self.update_gaussian_scale_t = False

            if type(self.focal_optimizer).__name__ == 'LBFGS':
                self.focal_optimizer.step(closure) # to use LBFGS
            else:
                self.focal_optimizer.step()


        for calib_id, cam_stack in self.calibration_groups.items():
            focal_delta = self.focal_delta_groups [ calib_id ].data.cpu().numpy()[0]
            for viewpoint_cam in cam_stack:
                focal = viewpoint_cam.fx
                viewpoint_cam.fx += focal_delta
                viewpoint_cam.fy += viewpoint_cam.aspect_ratio * focal_delta
            focal_grad  = self.focal_delta_groups [ calib_id ].grad.cpu().numpy()[0]
            print(f"\n\tfocal_update = {focal_delta},\tgradient = {focal_grad}")
            
            focal_grad_vec.append(focal_grad)
            focal_vec.append(focal)

        if self.num_line_elements > 0:
            self.focal_grad_stack.append(np.array(focal_grad_vec))
            self.focal_stack.append(np.array(focal_vec))


        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.00001:
        #     self.update_focal_learning_rate(lr=0.001)
        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.000001:
        #     self.update_focal_learning_rate(lr=0.0001)
        # if np.linalg.norm( np.array(focal_grad_vec) ) < 0.0000001:
        #     self.update_focal_learning_rate(lr=0.00001)
        converged = (np.linalg.norm( np.array(focal_grad_vec) ) < 0.0000001)
        return converged


    def kappa_step(self):
        self.__update_kappa_gradients()
        self.kappa_optimizer.step()
        for calib_id, cam_stack in self.calibration_groups.items():
            kappa_delta = self.kappa_delta_groups [ calib_id ].data.cpu().numpy()[0]
            for viewpoint_cam in cam_stack:
                viewpoint_cam.kappa += kappa_delta
            kappa_grad  = self.kappa_delta_groups [ calib_id ].grad.cpu().numpy()[0]
            print(f"\tkappa_update = {kappa_delta},\tgradient = {kappa_grad}")



    def zero_grad(self):
        for viewpoint_cam in self.viewpoint_stack:
            viewpoint_cam.cam_focal_delta.data.fill_(0)
            viewpoint_cam.cam_kappa_delta.data.fill_(0)
            if viewpoint_cam.cam_focal_delta.grad is not None:
                viewpoint_cam.cam_focal_delta.grad.detach_()
                viewpoint_cam.cam_focal_delta.grad.zero_()
            if viewpoint_cam.cam_kappa_delta.grad is not None:
                viewpoint_cam.cam_kappa_delta.grad.detach_()
                viewpoint_cam.cam_kappa_delta.grad.zero_()



    def update_focal_learning_rate (self, lr = None, scale = None):
        for param_group in self.focal_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr >= 0.01 else lr
        rich.print("\n[bold green]focal_optimizer.param_groups:[/bold green]", self.focal_optimizer.param_groups)



    def update_kappa_learning_rate (self, lr = None, scale = None):
        for param_group in self.kappa_optimizer.param_groups:
            if lr is not None:
                param_group["lr"] = lr
            if scale is not None:
                lr = param_group["lr"]
                param_group["lr"] = scale * lr if lr >= 0.001 else lr
        rich.print("\n[bold green]kappa_optimizer.param_groups:[/bold green]", self.kappa_optimizer.param_groups)



    def get_focal_statistics (self, all = False):
        if all:
            focal_grad_stack  = np.array(self.focal_grad_stack).transpose()
            focal_stack = np.array(self.focal_stack).transpose()
        else:
            focal_grad_stack  = np.array(self.focal_grad_stack[-self.num_line_elements:] ).transpose()
            focal_stack = np.array(self.focal_stack[-self.num_line_elements:]).transpose()
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



    def zero_grad(self):
        self.pose_optimizer.zero_grad()
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



