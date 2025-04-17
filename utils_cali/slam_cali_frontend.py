import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
# from utils_cali.camera_cali_utils import CameraForCalibration as Camera
from utils.camera_utils import Camera
from utils.eval_utils import save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

from optimizers import CalibrationOptimizer
from utils.slam_frontend import FrontEnd
from utils_cali.eval_cali_utils import eval_ate, save_gaussians_class, save_cali

import rich

import pickle
import os

class Simulator():
    def __init__(self, path):
        self.load_intrinsics(path)

    def load_intrinsics(self, path):
        self.fx = []
        self.fy = []
        self.cali_id = []
        i = 0
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines: 
                parts = line.split()
                focal = float(parts[0])
                if len(self.fx) != 0:
                    i += 1 if focal != self.fx[-1] else 0
                self.fx.append(focal)
                self.fy.append(focal)
                self.cali_id.append(i)

class FrontEndCali(FrontEnd):

    def __init__(self, config):
        super().__init__(config)

        # path = config["Dataset"]["intrinsic_path"]
        # if config["Dataset"]["intrinsic_path"] is not None:
        #     self.simulator = Simulator(config["Dataset"]["intrinsic_path"])
        # else:
        #     self.simulator = None
        path = config.get("Dataset", {}).get("intrinsic_filename", None)
        self.simulator = Simulator(config["Dataset"]["dataset_path"] + '/' + path) if path is not None else None
        self.use_gt_poses = False
        self.add_perterbation = False
    
    def tracking_use_gt_poses(self, viewpoint):
        viewpoint.R = viewpoint.R_gt
        viewpoint.T = viewpoint.T_gt
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        self.median_depth = get_median_depth(depth, opacity)

        return render_pkg


    def run(self):
        # assert self.dataset.num_imgs == self.simulator.fx.shape[0]
        self.MODULE_TEST_CALIBRATION = False
        print(f"self.MODULE_TEST_CALIBRATION: {self.MODULE_TEST_CALIBRATION}")
        print(f"self.signal_calibration_change: {self.signal_calibration_change}")
        cur_frame_idx = 0
        projection_matrix = None # projection_matrix is implemented as a property in Camera
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            # [i for i in range(0, self.dataset.num_imgs)], #when final frame is reached, evaluate the ATE of all frames
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                        save_gaussians_class(self.save_dir, self.gaussians)
                        save_cali(self.save_dir, self.cameras, self.kf_indices, self.dataset.num_imgs)

                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )

                viewpoint.compute_grad_mask(self.config)

                # if self.MODULE_TEST_CALIBRATION and self.simulator is not None:
                if self.simulator is not None:
                    viewpoint.calib_id = self.simulator.cali_id[cur_frame_idx]
                    focal_ref = None if viewpoint.calib_id == 0 else self.simulator.fx[cur_frame_idx]
                    viewpoint.fx_init = self.simulator.fx[cur_frame_idx]
                    viewpoint.fy_init = self.simulator.fy[cur_frame_idx]
                    viewpoint.kappa_init = 0.0 # backup
                    # viewpoint.fx = self.simulator.fx[cur_frame_idx]
                    # viewpoint.fy = self.simulator.fy[cur_frame_idx]
                if self.add_perterbation:
                    viewpoint.calib_id = 1
                    focal_per = self.config["Dataset"]["focal_perturbation"] if 'focal_perturbation' in self.config["Dataset"] else 1.01
                    viewpoint.fx = viewpoint.fx * focal_per
                    viewpoint.fy = viewpoint.fy * focal_per


                # initialize calibration and pose to the previous camera
                if len(self.cameras) > self.use_every_n_frames:
                    prev = self.cameras[cur_frame_idx - self.use_every_n_frames] # last frame in tracking
                    viewpoint.update_calibration (prev.fx, prev.fy, prev.kappa) # use last frame calibration

                    if self.use_gt_poses:
                        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt) # use last frame pose
                    else:
                        viewpoint.update_RT(prev.R, prev.T)

                    if viewpoint.calib_id != prev.calib_id:
                        if (not self.signal_calibration_change):
                            rich.print(f"\n[bold red]FrontEnd: calibration change detected at frame_idx: [/bold red]{cur_frame_idx}")
                            self.backend_queue.put(["calibration_change"])
                        self.signal_calibration_change = True
                    else:
                        self.signal_calibration_change = False

                if self.signal_calibration_change:
                    viewpoint.kappa = 0.0 # reset kappa to zero for new calibration
                    if self.requested_keyframe > 0:
                        time.sleep(0.01)
                        continue
                
                # rich.print(f"FrontEnd  Tracking t_gt: [{viewpoint.uid}]: t = {[f'{x.item():.8f}' for x in (viewpoint.T_gt)]}")
                # rich.print(f"FrontEnd  Tracking t   : [{viewpoint.uid}]: t = {[f'{x.item():.8f}' for x in (viewpoint.T)]}")
                
                
                # if self.MODULE_TEST_CALIBRATION and self.signal_calibration_change:
                #     if focal_ref is not None:
                #         rich.print(f"[bold magenta]At Frame {viewpoint.uid}, change focal length (fx) to: [/bold magenta] {focal_ref} ")
                #         viewpoint.fx = focal_ref
                #         viewpoint.fy = viewpoint.aspect_ratio * focal_ref

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )


                # focal tracking
                # if self.require_calibration and self.initialized and signal_calibration_change:
                #     self.init_focal (viewpoint, gaussian_scale_t = 10.0,  beta = 1.0, learning_rate = 0.1, max_iter_num = 20) #10% * 600 = 60
                #     self.init_focal (viewpoint, gaussian_scale_t = 0.0,  beta = 0.0, learning_rate = 0.01, max_iter_num = 50)
                # TUNING PARAMETERS
                if self.require_calibration and self.initialized and self.signal_calibration_change:
                    lr = self.init_focal (viewpoint, optimizer_type = "Adam", gaussian_scale_t = 10.0,  beta = 0.0, learning_rate = 0.1, max_iter_num = 30, step_safe_guard = False)
                    self.init_focal (viewpoint, optimizer_type = "SGD", gaussian_scale_t = 0.0,  beta = 1.0, learning_rate = lr, max_iter_num = 20, step_safe_guard = True)

                if self.use_gt_poses:
                    render_pkg = self.tracking_use_gt_poses(viewpoint)
                else:
                    render_pkg = self.tracking(cur_frame_idx, viewpoint)

                if self.require_calibration and self.initialized and self.signal_calibration_change:
                    self.init_focal (viewpoint, optimizer_type = "SGD", gaussian_scale_t = 0.0,  beta = 0.0, learning_rate = lr, max_iter_num = 20, step_safe_guard = True)

                # pose tracking
                # render_pkg = self.tracking(cur_frame_idx, viewpoint)
                # rich.print(f"FrontEnd  Tracking : [{cur_frame_idx}]: delta_t = {[f'{x.item():.8f}' for x in (viewpoint.T_gt - viewpoint.T)]}")




                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf: # or signal_calibration_change:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    rich.print(f"[bold blue]FrontEnd Send    :[/bold blue] [{cur_frame_idx}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calib_id}")

                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)
                    self.sync_backend_calibration(cur_frame_idx)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.sync_backend_calibration(cur_frame_idx)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
 

