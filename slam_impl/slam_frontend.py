import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from slam_impl.camera_utils import Camera
from slam_impl.eval_utils import eval_ate, save_gaussians
from slam_impl.logging_utils import Log
from slam_impl.multiprocessing_utils import clone_obj
from slam_impl.pose_utils import update_pose
from slam_impl.slam_utils import get_loss_tracking, get_median_depth, get_loss_tracking_no_grad_mask

from slam_impl.optimizers import CalibrationOptimizer, PoseOptimizer, lr_exp_decay_helper

from slam_impl.gaussian_scale_space import image_conv_gaussian
import copy
import rich
from PIL import Image

import matplotlib.pyplot as plt
import os
from gaussian_splatting.utils.system_utils import mkdir_p

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        # calibration control params
        self.require_calibration = False
        self.allow_lens_distortion = False
        self.MODULE_TEST_CALIBRATION = False
        self.signal_calibration_change = False
        self.calib_id = 0 # current calibration id
        self.calibration_frame_idx = 0   # when current calibration takes effect
        self.calibration_keyframe_sent = True
        self.calibration_initialized = True

        # use ground-truth poses for reconstruction, in which case poses will not be optimised
        self.use_gt_pose = config.get("use_gt_pose", False)

        # ATE array
        self.ATE_records = []


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint, focal_optimizer_type=None, learning_rate=0.001, grad_mask=True):

        # set to the ground truth pose
        if self.use_gt_pose:
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)  # use provided ground-truth pose
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
        
        # add calibration optimizer in tracking
        calibration_optimizers = None
        if focal_optimizer_type is not None:
            viewpoint_stack = [ viewpoint ]
            H = viewpoint.image_height
            W = viewpoint.image_width
            focal_ref = np.sqrt(H*H + W*W)/2
            calibration_optimizers = CalibrationOptimizer(viewpoint_stack, focal_ref, focal_optimizer_type= focal_optimizer_type)
            rich.print(f"[bold green]Initialize focal length optimizer: {focal_optimizer_type}, lr = {learning_rate} [/bold green]")
            calibration_optimizers.update_focal_learning_rate(lr = learning_rate)

        # prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        # viewpoint.update_RT(prev.R, prev.T)
        # print(f"len(self.cameras) = {len(self.cameras)},  cur_frame_idx = {cur_frame_idx},  self.use_every_n_frames = {self.use_every_n_frames}")
        # print(f"prev = {prev.uid},   viewpoint = {viewpoint.uid}")

        lr_scale_factor = 1.0 if calibration_optimizers is not None else 1.0
        tracking_itr_num = self.tracking_itr_num * 1 if calibration_optimizers is not None else self.tracking_itr_num

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                * lr_scale_factor,
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"]
                * lr_scale_factor,
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        if focal_optimizer_type is None:
            opt_params.append(
                {
                    "params": [viewpoint.exposure_a],
                    "lr": 0.01,
                    "name": "exposure_a_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.exposure_b],
                    "lr": 0.01,
                    "name": "exposure_b_{}".format(viewpoint.uid),
                }
            )
        

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            ) if grad_mask else get_loss_tracking_no_grad_mask (
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                if calibration_optimizers is not None:
                    calibration_optimizers.focal_step() # add update focal
                    # if self.allow_lens_distortion and tracking_itr > 10:
                    #     calibration_optimizers.kappa_step() # add update kappa
                    calibration_optimizers.zero_grad(set_to_none=True)
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T, kf_calib in keyframes:
            calib = copy.deepcopy(kf_calib)
            kf_fx, kf_fy, kf_kappa = calib[0], calib[1], calib[2]
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())
            self.cameras[kf_id].update_calibration(kf_fx, kf_fy, kf_kappa)



    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()


    def run(self):
        cur_frame_idx = 0
        # if self.dataset.focal_changed: # for simulated dataset, fx, fy, cx, cy, height, width are changing
        #     _,_,_,fx, fy, cx, cy, _, _, height, width, _ = self.dataset[cur_frame_idx]
        #     projection_matrix = getProjectionMatrix2(
        #         znear=0.01,
        #         zfar=100.0,
        #         fx=fx,
        #         fy=fy,
        #         cx=cx,
        #         cy=cy,
        #         W=width,
        #         H=height,
        #     ).transpose(0, 1)
        # else:
        #     projection_matrix = getProjectionMatrix2(
        #         znear=0.01,
        #         zfar=100.0,
        #         fx=self.dataset.fx,
        #         fy=self.dataset.fy,
        #         cx=self.dataset.cx,
        #         cy=self.dataset.cy,
        #         W=self.dataset.width,
        #         H=self.dataset.height,
        #     ).transpose(0, 1)
        # projection_matrix = projection_matrix.to(device=self.device)
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
                        ate = eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        self.ATE_records.append( {"kf_id": cur_frame_idx, "ate": ate} )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
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

 
                ###### test code block
                if self.MODULE_TEST_CALIBRATION:
                    if cur_frame_idx == 100:
                        viewpoint.calib_id = 1 # calibration change
                        focal_ref = 400
                    elif cur_frame_idx == 200:
                        viewpoint.calib_id = 1 # calibration change
                        focal_ref = 350
                    elif cur_frame_idx == 300:
                        viewpoint.calib_id = 1 # calibration change
                        focal_ref = 700
                    elif cur_frame_idx == 400:
                        viewpoint.calib_id = 1 # calibration change
                        focal_ref = 900
                    else:
                        viewpoint.calib_id = 0 # no calibration change
                        focal_ref = None
                    
                    self.save_trj_kf_intv = 1 if self.initialized else self.save_trj_kf_intv


                # the camera notifies the frontend calibration that there exists carlibation change by
                # passing a calib_id > 0
                # then frontend reset the right calibration identifier, by accumulating on the local calib_id
                if viewpoint.calib_id > 0:  # expected value: 0, 1                    
                    if (not self.signal_calibration_change): # only do it once
                        self.calib_id += viewpoint.calib_id
                        self.calibration_frame_idx = cur_frame_idx                        
                        rich.print(f"\n[bold red]FrontEnd: calibration change detected at frame_idx: [/bold red]{cur_frame_idx:05d}")
                        self.backend_queue.put(["calibration_change"])
                    self.signal_calibration_change = True
                    self.calibration_keyframe_sent = False

                else:
                    self.signal_calibration_change = False
                viewpoint.calib_id = self.calib_id

                if (not self.reset):
                    prev = self.cameras[cur_frame_idx - self.use_every_n_frames] # last frame in tracking
                    viewpoint.update_calibration (prev.fx, prev.fy, prev.kappa) # use last frame calibration
                    viewpoint.update_RT(prev.R, prev.T) # use last frame pose

                # if self.signal_calibration_change:
                #     if self.requested_keyframe > 0:
                #         time.sleep(0.01)
                #         continue


                ###### test code block
                if self.MODULE_TEST_CALIBRATION and self.signal_calibration_change:
                    if focal_ref is not None:
                        rich.print(f"[bold magenta]At Frame {viewpoint.uid}, change focal length (fx) to: [/bold magenta] {focal_ref} ")
                        viewpoint.fx = focal_ref
                        viewpoint.fy = viewpoint.aspect_ratio * focal_ref


                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )


                # TUNING PARAMETERS
                if self.require_calibration and self.initialized and self.signal_calibration_change:
                    """
                    Focal-length initialization
                    """

                    self.calibration_initialized = False

                    save_info = "frame"+str(cur_frame_idx)

                    w, h = viewpoint.image_width, viewpoint.image_height
                    scale_t = 0.01 * max(w,h)

                    """
                    at coarse scale, scale_t decided by image size
                    """
                    # Adam+SGD, at the same scale
                    lr = self.init_focal (viewpoint, optimizer_type = "Adam", image_grad_mask=False, gaussian_scale_t = scale_t,  learning_rate = 0.1,  max_iter_num = 30, step_safe_guard = False, save_info=save_info)
                    lr = min(lr, 1.0) # safe-guard
                    _  = self.init_focal (viewpoint, optimizer_type = "SGD",  image_grad_mask=False, gaussian_scale_t = scale_t,  learning_rate = lr,    max_iter_num = 20, step_safe_guard = True )

                    """
                    at scale 0
                    """
                    # focal should be close to ground-truth now, but not accurate if the pose changes a lot
                    _  = self.init_focal (viewpoint, optimizer_type = "SGD",  image_grad_mask=False,  gaussian_scale_t = 0.0,  learning_rate = 0.1, max_iter_num = 20, step_safe_guard = True )
                    
                if (not self.calibration_keyframe_sent):
                    """
                    Pose initialization
                    Pose and focal-length joint refinement
                    """
                    frontend_strategy = 3

                    if frontend_strategy == 1:

                        # ATE: 0.03172540502845
                        render_pkg = self.tracking(cur_frame_idx, viewpoint)
                        _  = self.init_focal (viewpoint, optimizer_type = "SGD",  image_grad_mask=False,  gaussian_scale_t = 0.0,  learning_rate = 0.1, max_iter_num = 10, step_safe_guard = True )
                        render_pkg = self.tracking(cur_frame_idx, viewpoint, focal_optimizer_type = "SGD",  learning_rate=0.001, grad_mask=True)

                    elif frontend_strategy == 2:

                        render_pkg = self.tracking(cur_frame_idx, viewpoint, focal_optimizer_type = "SGD",  learning_rate=0.001, grad_mask=True)
                        _  = self.init_focal (viewpoint, optimizer_type = "SGD",  image_grad_mask=True,  gaussian_scale_t = 0.0,  learning_rate = 0.1, max_iter_num = 10, step_safe_guard = True )

                    elif frontend_strategy == 3:

                        render_pkg = self.tracking(cur_frame_idx, viewpoint)
                        render_pkg = self.tracking(cur_frame_idx, viewpoint, focal_optimizer_type = "SGD",  learning_rate=0.001, grad_mask=True)
                        _  = self.init_focal (viewpoint, optimizer_type = "SGD",  image_grad_mask=True,  gaussian_scale_t = 0.0,  learning_rate = 0.1, max_iter_num = 10, step_safe_guard = True )


                else:
                    render_pkg = self.tracking(cur_frame_idx, viewpoint)

                plt.close()

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
                    Log(f"Frontend cannot send frame: {cur_frame_idx=}, becuase {self.requested_keyframe=}")
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
                if create_kf: #or self.signal_calibration_change:                    
                    # removed = None
                    # if (not create_kf) and self.signal_calibration_change: # if not a keyframe, but calibration changes
                    #     self.current_window[0] = cur_frame_idx # replace the last keyframe with the current keyframe
                    #     removed = [0]
                    # else:
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
                    self.calibration_keyframe_sent = True
                    CC = viewpoint.camera_center.cpu().numpy()
                    exposure_a = viewpoint.exposure_a.data.item()
                    exposure_b = viewpoint.exposure_b.data.item()
                    rich.print(f"[bold blue]FrontEnd Send    :[/bold blue] [{cur_frame_idx:05d}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calib_id}. cam_center: ({CC[0]:.3f}, {CC[1]:.3f}, {CC[2]:.3f}), exposure: (a: {exposure_a:.5f}, b: {exposure_b:.5f})")
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
                    ate = eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                    self.ATE_records.append( {"kf_id": cur_frame_idx, "ate": ate} )
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

                elif data[0] == "update_calibration":
                    calib_id = data[1]
                    kf_calib = data[2]
                    self.calibration_initialized = data[3]
                    if self.calib_id == calib_id:
                        calib = copy.deepcopy(kf_calib)
                        kf_fx, kf_fy, kf_kappa = calib[0], calib[1], calib[2]
                        self.cameras[cur_frame_idx-self.use_every_n_frames].update_calibration(kf_fx, kf_fy, kf_kappa)
                        rich.print(f"[bold blue]FrontEnd Recieve :[/bold blue] [{cur_frame_idx:05d}]: update_calibration: fx: {kf_fx:.3f}, fy: {kf_fy:.3f}, kappa: {kf_kappa:.6f}, calib_id: {calib_id}, initialized: {self.calibration_initialized}")



    
    def init_focal (self, viewpoint, optimizer_type = "Adam", image_grad_mask=False, gaussian_scale_t = 5.0, learning_rate = 0.1, max_iter_num = 20, step_safe_guard = False, save_info=None):

        viewpoint_stack = []
        viewpoint_stack.append(viewpoint)

        H = viewpoint.image_height
        W = viewpoint.image_width
        focal_ref = np.sqrt(H*H + W*W)/2

        calibration_optimizers = CalibrationOptimizer(viewpoint_stack, focal_ref, focal_optimizer_type= optimizer_type) # only one view
        calibration_optimizers.num_line_elements = max_iter_num # sample points for line fitting
        rich.print(f"[bold green]Initialize focal length optimizer: {optimizer_type}, lr = {learning_rate} [/bold green]")
        calibration_optimizers.update_focal_learning_rate(lr = learning_rate)

        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]

        gt_image = viewpoint.original_image.cuda()
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        if image_grad_mask:
            rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask # don't apply gradient mask with Gaussian scale space

        # Gaussian scale space
        gt_image_scale_t = image_conv_gaussian(gt_image, sigma=gaussian_scale_t, epsilon=0.01) if gaussian_scale_t > 0.5 else gt_image


        if save_info is not None:
            img_dir = os.path.join(self.save_dir, "images", str(save_info))
            mkdir_p(img_dir)
            self.save_tensor2rgb(gt_image, os.path.join(img_dir, "gt_image.png") )
            self.save_tensor2rgb(gt_image_scale_t, os.path.join(img_dir, "gt_image_scale_t.png") )
            self.save_tensor2rgb(rgb_pixel_mask, os.path.join(img_dir, "rgb_pixel_mask.png") )
            self.save_tensor2rgb(viewpoint.grad_mask, os.path.join(img_dir, "viewpoint_grad_mask.png") )


        loss_prev = 1e10

        for itr in range(max_iter_num):

            if itr % 5 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            # not really necessary
            image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
     
            # Gaussian scale space
            image_scale_t = image_conv_gaussian(image_ab, sigma=gaussian_scale_t, epsilon=0.01) if gaussian_scale_t > 0.5 else image_ab


            use_smooth_l1 = (gaussian_scale_t > 0.5)
            use_smooth_l1 = False # previously set to false all the time

            if use_smooth_l1:
                """
                Use a Huber-type loss function for smooth gradients at minumum
                - HuberLoss
                - SmoothL1Loss
                parameters decided by residual = |f(x) - y|
                """
                mask = opacity * rgb_pixel_mask
                huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = 0.001) # or 0.001. use small enough beta for smooth gradient close to groud-truth
                loss = huber_loss_function(image_scale_t*mask, gt_image_scale_t*mask)

            else:
                """
                standard L1 loss
                """
                l1 = opacity * rgb_pixel_mask * torch.abs(image_scale_t - gt_image_scale_t)
                loss = l1.mean()


            if save_info is not None:
                postfix = "_scale"+str(gaussian_scale_t) + "_itr"+str(itr)+"_focal"+str(viewpoint.fx)+".png"
                img_dir = os.path.join(self.save_dir, "images", str(save_info))
                self.save_tensor2rgb(image, os.path.join(img_dir, "image"+postfix) )
                # self.save_tensor2rgb(image_ab, os.path.join(img_dir, "image_ab"+postfix) )
                self.save_tensor2rgb(image_scale_t, os.path.join(img_dir, "image_scale_t"+postfix) )
                self.save_tensor2rgb(opacity, os.path.join(img_dir, "opacity"+postfix) )

            
            # print(f"focal_init: iter: [{itr}]")
            if step_safe_guard and (loss > loss_prev):
                rich.print(f"[bold yellow][Warning]: loss={loss:.8f}, loss_prev={loss_prev:.8f}. revoke previous step and shrink learning rate[/bold yellow]")
                # print(f"loss_prev = {loss_prev},   loss = {loss},   current_fx = {viewpoint.fx}")
                calibration_optimizers.undo_focal_step()
                # print(f"\t after revoling, current_fx = {viewpoint.fx}")
                calibration_optimizers.update_focal_learning_rate(scale=0.5)
                with torch.no_grad():
                    calibration_optimizers.focal_step() # step again with old gradient
                # print(f"\t update to,      current_fx = {viewpoint.fx}\n")
                continue

            if loss < loss_prev:
                loss_prev = loss

            # clear old gradient, and compute new gradient
            with torch.no_grad():
                calibration_optimizers.zero_grad(set_to_none=True)
            loss.backward()
            
            with torch.no_grad():
                converged = calibration_optimizers.focal_step() # optimize focal only
                if converged:
                    break

        return calibration_optimizers.estimate_step_size()
    

    def sync_backend_calibration (self, cur_frame_idx):
        if len(self.current_window):
            last_keyframe_idx = self.current_window[0]
            if last_keyframe_idx < self.calibration_frame_idx or self.calibration_frame_idx == 0:
                return
            # print(f"{last_keyframe_idx=}")
            # print(f"{self.calibration_frame_idx=}")
            # print(f"{cur_frame_idx=}")
            last_keyframe = self.cameras[last_keyframe_idx] # last keyframe (optimzied by backend)
            # last_frame = self.cameras[cur_frame_idx - self.use_every_n_frames] # last frame in tracking
            kf_calib = copy.deepcopy( [last_keyframe.fx, last_keyframe.fy, last_keyframe.kappa] )
            for frame_idx in range(self.calibration_frame_idx, cur_frame_idx, self.use_every_n_frames):
                frame = self.cameras[frame_idx]
                assert frame.calib_id == last_keyframe.calib_id, f"{frame.calib_id=}\t{last_keyframe.calib_id=}"
                # if frame.calib_id == last_keyframe.calib_id:
                calib = kf_calib.copy()
                kf_fx, kf_fy, kf_kappa = calib[0], calib[1], calib[2]
                frame.update_calibration (kf_fx, kf_fy, kf_kappa)
    


    @staticmethod
    def save_tensor2rgb (torch_tensor, filename="test.png"):
        image = torch_tensor.squeeze().squeeze()
        if not (image.ndim == 2 or image.ndim == 3):
            return
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        '''
            Torch Image: 3*M*N
            Numpy Image: M*N*3 [RGB]
            OpenCV stores images in BGR order instead of RGB
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        '''
        image = torch.clamp(image, min=0, max=1.0) * 255
        rgb = image.byte().permute(1, 2, 0).contiguous().cpu().numpy()        
        Image.fromarray(rgb).save(filename)
        # plt.imsave(filename, rgb)

