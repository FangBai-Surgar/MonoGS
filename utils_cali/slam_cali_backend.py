import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.general_utils import helper
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping

from optimizers import CalibrationOptimizer, PoseOptimizer, lr_exp_decay_helper
import numpy as np
import rich
from utils.slam_backend import BackEnd
from utils_cali.eval_cali_utils import eval_cali



class BackEndCali(BackEnd):
    def __init__(self, config):
        super().__init__(config)
        self.use_gt_poses = False

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, calibrate=False, iters=10)
                    self.push_to_frontend()
                    rich.print("[bold yellow]Backend : no data from front-end, continue optimizing existing data [/bold yellow]")
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "calibration_change":
                    self.map(self.current_window, prune=True, calibrate=False, iters=10)
                    self.map(self.current_window, prune=False, calibrate=False, iters=10)
                    self.map(self.current_window, prune=True, calibrate=False, iters=1)
                    self.push_to_frontend()
                    rich.print("[bold red]Backend : calibration change signal recieved [/bold red]")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    rich.print(f"[bold blue]BackEnd  Receive :[/bold blue] [{cur_frame_idx}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calibration_identifier}")

                    current_calibration_identifier = viewpoint.calibration_identifier
                    # calibration_identifier_cnt = 0

                    if len(self.current_window):
                        last_keyframe = self.viewpoints[ self.current_window[0] ]
                        if (current_calibration_identifier == last_keyframe.calibration_identifier):
                            viewpoint.update_calibration(last_keyframe.fx, last_keyframe.fy, last_keyframe.kappa) # use the calibration estimate in backend keyframes
                            self.signal_calibration_change = False
                        else:
                            self.signal_calibration_change = True

                    if self.signal_calibration_change:
                        self.calibration_initialized = False
                        self.calibration_identifier_cnt = 0
                    self.calibration_identifier_cnt += 1

                    rich.print(f"[bold blue]BackEnd  InitEst :[/bold blue] [{cur_frame_idx}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calibration_identifier}")

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    # cali_id doesn't change OR rgbd mode
                    # if (not self.signal_calibration_change) or (self.config["Training"]["sensor_type"] == 'depth'):
                    if (not self.signal_calibration_change):
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                        rich.print(f"[bold blue]BackEnd  AddKF depth_map:[/bold blue] [{cur_frame_idx}]")               
                    else:
                        pass
                        # new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
                        # new_n_obs = torch.zeros((new_xyz.shape[0])).int()
                        # if new_kf_ids is not None:
                        #     self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
                        # if new_n_obs is not None:
                        #     self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()                 
                    
                    pose_opt_params = []
                    calib_opt_frames_stack = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("after_mapping_itr_num", 30)
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num                    
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            if not self.use_gt_poses:
                                pose_opt_params.append(
                                    {
                                        "params": [viewpoint.cam_rot_delta],
                                        "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                        * 0.5,
                                        "name": "rot_{}".format(viewpoint.uid),
                                    }
                                )
                                pose_opt_params.append(
                                    {
                                        "params": [viewpoint.cam_trans_delta],
                                        "lr": self.config["Training"]["lr"][
                                            "cam_trans_delta"
                                        ]
                                        * 0.5,
                                        "name": "trans_{}".format(viewpoint.uid),
                                    }
                                )
                            calib_opt_frames_stack.append(viewpoint)
                            # calibration_identifier_cnt += 1 if viewpoint.calibration_identifier == current_calibration_identifier else 0
                        if not self.use_gt_poses:
                            pose_opt_params.append(
                                {
                                    "params": [viewpoint.exposure_a],
                                    "lr": 0.01,
                                    "name": "exposure_a_{}".format(viewpoint.uid),
                                }
                            )
                            pose_opt_params.append(
                                {
                                    "params": [viewpoint.exposure_b],
                                    "lr": 0.01,
                                    "name": "exposure_b_{}".format(viewpoint.uid),
                                }
                            )
                    if not self.use_gt_poses:
                        self.keyframe_optimizers = torch.optim.Adam(pose_opt_params)
                        self.keyframe_optimizers.zero_grad()

                    
                    # if self.require_calibration and self.initialized and calibration_identifier_cnt >= 1 and current_calibration_identifier != 0:
                    # and window is full
                    if self.require_calibration and self.initialized and self.calibration_identifier_cnt >= 1 and current_calibration_identifier != 0 and (not self.calibration_initialized):

                    # if self.require_calibration and self.initialized and calibration_identifier_cnt >= 1 and current_calibration_identifier != 0 and len(self.current_window) == self.config["Training"]["window_size"]:
                        # self.viewpoint_refinement(self.current_window, iters=50)
                        H = viewpoint.image_height
                        W = viewpoint.image_width
                        focal_ref = np.sqrt(H*H + W*W)/2
                        rich.print("[bold green]calibration optimizer[/bold green]. current_window: ", current_window)    
                        self.calibration_optimizers = CalibrationOptimizer(calib_opt_frames_stack, focal_ref, focal_optimizer_type="Adam")
                        # self.calibration_optimizers = CalibrationOptimizer(calib_opt_frames_stack, focal_ref, focal_optimizer_type="SGD")
                        self.calibration_optimizers.num_line_elements = 0 # sample points for line fitting
                    else:
                        self.calibration_optimizers = None

                    iters = int(iter_per_kf/2) if self.calibration_optimizers is not None else iter_per_kf
                    # TODO: add decay learning rate + SGD

                    ### The order of following three matters a lot! ###
                    if self.calibration_optimizers is not None:
                        if (self.calibration_identifier_cnt == 1): # Don't update 3D structure with one view
                            rich.print("[bold green]cali_id_cnt == 1[/bold green]")
                            lr1 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt1", 0.002)
                            self.calibration_optimizers.update_focal_learning_rate(lr = lr1)
                            self.map(self.current_window, calibrate=True, fix_gaussian=True,  iters=iter_per_kf*3)

                        elif (self.calibration_identifier_cnt == 2):
                            rich.print("[bold green]cali_id_cnt == 2[/bold green]")
                            lr2 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt2", 0.002)
                            self.calibration_optimizers.update_focal_learning_rate(lr = lr2)
                            self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf*2) # more iters for two views
                            # self.map(self.current_window, prune=True, iters=5)
                            # afle = eval_cali(self.viewpoints, None)
                            # rich.print(f"[bold blue]BackEnd  AFLE:[/bold blue] [{cur_frame_idx}]: {afle:.6f}\n")
                        elif (self.calibration_identifier_cnt == len(self.current_window)):
                            lr2 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt2", 0.002)
                            self.calibration_optimizers.update_focal_learning_rate(lr = lr2)
                            self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf*2) # BA with full window

                        else:
                            # rich.print("[bold green]cali_id_cnt != 1 and != 2[/bold green]")
                            lr2 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt2", 0.002)
                            # lr = helper(self.counter, lr2, 0.001, lr_delay_steps=2, lr_delay_mult=0.1, max_steps=1000000)
                            self.calibration_optimizers.update_focal_learning_rate(lr = lr2)
                            # test 1 + rgbd + add kf at first + adam -> afle = 7
                            # test 1 + rgbd + add kf at last + adam -> afle = 
                            # test 1 + rgbd + add kf at last + sgd -> afle = 6.8
                            self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf)
                            # test 2 + rgbd + add kf at first -> afle = 10
                            # test 2 + rgbd + add kf at last -> afle = 13
                            # self.map(self.current_window, calibrate=False, fix_gaussian=True, iters=iter_per_kf)
                            # self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf)
                            # self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf)
                            # test 3 + rgbd + add kf at last + sgd + decay -> afle = 
                            # self.map(self.current_window, calibrate=True, fix_gaussian=False, iters=iter_per_kf)


                            # afle = eval_cali(self.viewpoints, None)
                            # rich.print(f"[bold blue]BackEnd  AFLE:[/bold blue] [{cur_frame_idx}]: {afle:.6f}\n")
                    else:
                        self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)


                    if self.calibration_identifier_cnt == 10: # number of keyframes after calibration change
                        self.calibration_initialized = True

                    # update all cameras with the most recent calibration_identifier
                    if self.calibration_optimizers is not None:
                        fx = self.viewpoints[cur_frame_idx].fx
                        fy = self.viewpoints[cur_frame_idx].fy
                        kappa = self.viewpoints[cur_frame_idx].kappa
                        for cam_id, viewpoint in self.viewpoints.items():
                            if viewpoint.calibration_identifier == current_calibration_identifier:
                                viewpoint.update_calibration(fx, fy, kappa)
                    rich.print(f"[bold blue]BackEnd  Optimize:[/bold blue] [{cur_frame_idx}]: fx: {self.viewpoints[cur_frame_idx].fx:.3f}, fy: {self.viewpoints[cur_frame_idx].fy:.3f}, kappa: {self.viewpoints[cur_frame_idx].kappa:.6f}, calib_id: {self.viewpoints[cur_frame_idx].calibration_identifier}, iter_per_kf: {iter_per_kf}\n")
                    
                    # if len(self.viewpoints) > 1:
                    #     # ate = backend_eval(self.viewpoints, None)
                    #     afle = eval_cali(self.viewpoints, None)
                    #     # rich.print(f"[bold blue]BackEnd  ATE:[/bold blue] [{cur_frame_idx}]: {ate:.6f}\n")
                    #     rich.print(f"[bold blue]BackEnd  AFLE:[/bold blue] [{cur_frame_idx}]: {afle:.6f}\n")
                    
                    rich.print(f"[bold blue]BackEnd  Optimize:[/bold blue] [{cur_frame_idx}]: fx: {self.viewpoints[cur_frame_idx].fx:.3f}, fy: {self.viewpoints[cur_frame_idx].fy:.3f}, kappa: {self.viewpoints[cur_frame_idx].kappa:.6f}, calib_id: {self.viewpoints[cur_frame_idx].calibration_identifier}, iter_per_kf: {iter_per_kf}\n")
                    self.push_to_frontend("keyframe")

                    # add depth points at last, because these points will not be optimized with one view.
                    if (self.signal_calibration_change):
                        self.add_next_kf(cur_frame_idx, self.viewpoints[cur_frame_idx], depth_map=depth_map) 
                        self.map(self.current_window, calibrate=False, iters=iter_per_kf) # don't calibrate with one view. optimize gaussian
                        # self.map(self.current_window, prune=True)
                    
                    
                    # temp = self.gaussians.copy()

                else:
                    raise Exception("Unprocessed data", data)
        
        self.save_calib_results()

        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return