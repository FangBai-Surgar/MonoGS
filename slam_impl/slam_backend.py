import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from slam_impl.logging_utils import Log
from slam_impl.multiprocessing_utils import clone_obj
from slam_impl.pose_utils import update_pose
from slam_impl.slam_utils import get_loss_mapping, get_loss_tracking, get_median_depth

from slam_impl.optimizers import CalibrationOptimizer, PoseOptimizer, lr_exp_decay_helper
import numpy as np
import copy
import rich


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.calibration_optimizers = None

        # calibration control params
        self.require_calibration = False
        self.allow_lens_distortion = False
        self.signal_calibration_change = False
        self.calibration_initialized = True
        self.calibration_keyframe_idx = 0
        self.calibration_window = []

        # use ground-truth poses for reconstruction, in which case poses will not be optimised
        self.use_gt_pose = config.get("use_gt_pose", False)


    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.lr_cnt1 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt1", 0.002) # Adam
        self.lr_cnt2 = self.config.get("Dataset", {}).get("SelfCalibration", {}).get("backend_params", {}).get("lr_cnt2", 0.001) # SGD


    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.calibration_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, calibrate=0, fix_gaussian = False, iters=1):
        if len(current_window) == 0:
            return

        self.keyframe_optimizers.zero_grad(set_to_none=True)
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.calibration_optimizers is not None:
            self.calibration_optimizers.zero_grad(set_to_none=True)

        if fix_gaussian:
            self.map_fix_gaussian (current_window=current_window, calibrate=calibrate, iters=iters)
            return False

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for cur_itr in range(iters):
            if not fix_gaussian:
                self.iteration_count += 1            
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean() # if (not self.use_gt_pose) else 0
            # loss_mapping += 0.01*self.gaussians.get_opacity.mean() if calibrate else 0 #  # loss to enhance sparsity
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune and (not fix_gaussian):
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                """
                3DGS Dynamic control strategy
                """
                if (not fix_gaussian):

                    for idx in range(len(viewspace_point_tensor_acm)):
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                            radii_acm[idx][visibility_filter_acm[idx]],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                        )
                    update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                    )
                    if ( update_gaussian ):
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                        )
                        Log("gaussians.densify_and_prune")
                        gaussian_split = True

                    ## Opacity reset
                    if (self.iteration_count % self.gaussian_reset) == 0 and (not fix_gaussian) and (
                        not update_gaussian
                    ):
                        Log("Resetting the opacity of non-visible Gaussians")
                        self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                        gaussian_split = True


                # Calibration update. only do calibration if slam has been initialized.
                if calibrate and self.require_calibration and self.initialized:
                    if (self.calibration_optimizers is not None) and (not prune) and (not gaussian_split):
                        self.calibration_optimizers.focal_step()
                        if self.allow_lens_distortion and cur_itr > 5:
                            self.calibration_optimizers.kappa_step()
                if self.calibration_optimizers is not None:
                    self.calibration_optimizers.zero_grad(set_to_none=True)

                # Pose update
                if not self.use_gt_pose:
                    self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                """
                idea: don't update poses with a different calib_id, i.e., those poses before calibration.
                It seems there is no reason to do so, as we may fix Gaussians directly?
                """
                if not self.use_gt_pose:
                    if (not self.calibration_initialized) and calibrate==-1: # calibration with one view
                        for cam_idx in range( len(current_window) ):
                            viewpoint = viewpoint_stack[cam_idx]
                            if viewpoint.uid == 0:
                                continue
                            # only update frames with new calibration id
                            if current_window[cam_idx] < self.calibration_keyframe_idx:
                                continue
                            update_pose(viewpoint)
                    else:
                        # original pose update in GS-SLAM
                        for cam_idx in range(min(frames_to_optimize, len(current_window))):
                            viewpoint = viewpoint_stack[cam_idx]
                            if viewpoint.uid == 0:
                                continue
                            update_pose(viewpoint)


                # Structure (3D Gaussian) update
                if not fix_gaussian:
                    self.gaussians.optimizer.step()
                    self.gaussians.update_learning_rate(self.iteration_count)
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                

        return gaussian_split
    


    def map_fix_gaussian (self, current_window, calibrate=0, iters=1):
        # print(f"\n@run map_fix_gaussian  {current_window=}, {calibrate=}, {iters=}")
        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        frames_to_optimize = self.config["Training"]["pose_window"]
        # frames_to_optimize = min(frames_to_optimize, calibrate)

        for cur_itr in range(iters):      
            self.last_sent += 1

            loss_mapping = 0
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                # use tracking loss here
                loss_mapping += get_loss_tracking(
                                self.config, image, depth, opacity, viewpoint
                )
                # loss_mapping += get_loss_mapping(
                #     self.config, image, depth, viewpoint, opacity
                # )
            loss_mapping.backward()

            with torch.no_grad():

                if calibrate and self.require_calibration and self.initialized:
                    if (self.calibration_optimizers is not None):
                        self.calibration_optimizers.focal_step()
                        if self.allow_lens_distortion and cur_itr > 5:
                            self.calibration_optimizers.kappa_step()
                if self.calibration_optimizers is not None:
                    self.calibration_optimizers.zero_grad(set_to_none=True)
                
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

                self.gaussians.optimizer.zero_grad(set_to_none=True)
        return



    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000 if not self.use_gt_pose else 260000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")
        return

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            kf_calib = copy.deepcopy([kf.fx, kf.fy, kf.kappa])
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone(), kf_calib))
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)


    def save_calib_results (self):
        print(f"\n\nCalibration results")
        for cam_id, viewpoint in self.viewpoints.items():
            print(f"cam_id: {cam_id}: \tcalib_id: {viewpoint.calib_id}: fx = {viewpoint.fx:.3f}, fy = {viewpoint.fy:.3f}, kappa = {viewpoint.kappa:.6f}")        
        return
    



    def prune_floaters(self, min_opacity, extent, max_screen_size): 
        mask = self.gaussians.unique_kfIDs >= self.calibration_keyframe_idx
        prune_mask = (self.gaussians.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.gaussians.max_radii2D > max_screen_size
            big_points_ws = self.gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        prune_mask = torch.logical_and( prune_mask, mask.cuda())
        self.gaussians.prune_points(prune_mask)



    def create_rendered_depthmap (self, viewpoint):
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        (
            image,
            depth,
            opacity
        ) = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"]
        )

        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        depth = depth.detach().clone()
        opacity = opacity.detach()
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

                """
                operations to peform when awaiting frontend
                """
                if len(self.calibration_window):
                    # if len(self.calibration_window) == 1 only, this allows two-view Gaussian optimization before adding the third view
                    fix_gaussian= ( len(self.calibration_window) == 1 or len(self.calibration_window) == 2)
                    # fix_gaussian= ( len(self.calibration_window) == 1 )
                    self.map(self.current_window, calibrate=len(self.calibration_window), fix_gaussian=fix_gaussian)
                else:
                    self.map(self.current_window)
                                
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
                    rich.print("[bold yellow]Backend : no data from front-end, continue optimizing existing data [/bold yellow]")
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    # self.save_calib_results()
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
                    rich.print("[bold red]Backend : calibration change signal recieved [/bold red]")                    
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    self.map(self.current_window, iters=10 )
                    self.map(self.current_window, prune=True, iters=1)
                    self.push_to_frontend()
                    rich.print("[bold red]Backend : calibration change signal processed [/bold red]")   

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    rich.print(f"[bold blue]BackEnd  Receive :[/bold blue] [{cur_frame_idx:05d}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calib_id}")

                    current_calib_id = viewpoint.calib_id                  

                    if len(self.current_window):
                        last_keyframe = self.viewpoints[ self.current_window[0] ]
                        if (current_calib_id == last_keyframe.calib_id):
                            viewpoint.update_calibration(last_keyframe.fx, last_keyframe.fy, last_keyframe.kappa) # use the calibration estimate in backend keyframes
                            self.signal_calibration_change = False
                        else:
                            self.signal_calibration_change = True

                    if self.signal_calibration_change:
                        self.calibration_keyframe_idx = cur_frame_idx
                        self.calibration_initialized = False


                    if self.require_calibration and self.initialized and (not self.calibration_initialized):
                        self.calibration_window.append(cur_frame_idx)


                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    """
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    move backwards
                    insert new Gaussians after calibration
                    """

                    pose_opt_params = []
                    calib_opt_frames_stack = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf_multithread = self.config["Training"]["after_mapping_itr_num"] if "after_mapping_itr_num" in self.config["Training"].keys() else 10
                    # iter_per_kf = self.mapping_itr_num if self.single_thread else iter_per_kf_multithread
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
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

                        if (self.calibration_initialized) or (viewpoint.calib_id != current_calib_id):
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


                    self.keyframe_optimizers = torch.optim.Adam(pose_opt_params)
                    self.keyframe_optimizers.zero_grad()
                    self.calibration_optimizers = None

                    self.gaussians.optimizer.zero_grad()


                    """
                    Add new points from depth map
                    """
                    cur_keyframe = self.viewpoints[cur_frame_idx]

                    if (not self.monocular) or self.calibration_initialized:
                        self.add_next_kf(cur_frame_idx, cur_keyframe, depth_map=depth_map) 


                    if self.calibration_initialized and len(self.calibration_window):
                        self.calibration_window.clear()
                        Log("Calibration Initialized")

                    """
                    Uncalibrated Dense Bundle Adjustment (pose, Gaussians, calibration)

                    In the monocular case, we can rerender depth map, however we need to think about when to add new points
                    """
                    if len(self.calibration_window) > 0:


                        n_view_calib = 3
                        frames_to_optimize = self.config["Training"]["pose_window"]
                        n_view_calib = min(n_view_calib, frames_to_optimize)

                        H = cur_keyframe.image_height
                        W = cur_keyframe.image_width
                        focal_ref = np.sqrt(H*H + W*W)/2
                        window_id_cnt = sum(i >= self.calibration_keyframe_idx for i in self.current_window)                       
                        rich.print(f"[bold green]calibration optimizer[/bold green]:\n\tcurrent_window: {self.current_window}\n\tcalibration_window: {self.calibration_window}\n\tcalibration_keyframe_idx: {self.calibration_keyframe_idx}\n\tno. calibration views in window: {window_id_cnt}/{len(self.calibration_window)}")
                        
                        # number of keyframes after calibration change
                        if (len(self.calibration_window) == n_view_calib):

                            if self.monocular:
                                for cam_id in self.calibration_window:
                                    viewpoint = self.viewpoints[cam_id]
                                    depth_map = self.create_rendered_depthmap(viewpoint)
                                    self.add_next_kf(cam_id, viewpoint, depth_map=depth_map)

                            # if self.monocular:
                            #     depth_map = self.create_rendered_depthmap(cur_keyframe)
                            #     self.add_next_kf(cur_frame_idx, cur_keyframe, depth_map=depth_map)

                            self.calibration_optimizers = CalibrationOptimizer(calib_opt_frames_stack, focal_ref, focal_optimizer_type="Adam") 
                            self.calibration_optimizers.update_focal_learning_rate(lr = 0.002)
                            self.map(self.current_window, calibrate=len(self.calibration_window), iters=iter_per_kf)

                            
                            self.calibration_initialized = True
                            


                        elif (len(self.calibration_window) == 1):

                            self.calibration_optimizers = CalibrationOptimizer(calib_opt_frames_stack, focal_ref, focal_optimizer_type="Adam") 
                            self.calibration_optimizers.update_focal_learning_rate(lr = 0.002)
                            self.map(self.current_window, calibrate=len(self.calibration_window), iters=iter_per_kf, fix_gaussian=True)

                        elif (len(self.calibration_window) == 2):

                            self.calibration_optimizers = CalibrationOptimizer(calib_opt_frames_stack, focal_ref, focal_optimizer_type="Adam") 
                            self.calibration_optimizers.update_focal_learning_rate(lr = 0.002)
                            self.map(self.current_window, calibrate=len(self.calibration_window), iters=iter_per_kf, fix_gaussian=True)




                        # """
                        # push new calibration to frontend immediately
                        # """
                        # cur_keyframe = self.viewpoints[cur_frame_idx]
                        kf_calib = copy.deepcopy( [cur_keyframe.fx, cur_keyframe.fy, cur_keyframe.kappa] )
                        # msg = ["update_calibration", cur_keyframe.calib_id, kf_calib, self.calibration_initialized]
                        # self.frontend_queue.put(msg)

                        # update all cameras with the most recent calib_id
                        if self.calibration_optimizers is not None:
                            for cam_id in self.calibration_window:
                                viewpoint = self.viewpoints[cam_id]
                                assert cam_id >= self.calibration_keyframe_idx
                                assert viewpoint.calib_id == current_calib_id, f"slam_backend. calib_id mismatch: {viewpoint.calib_id=}\t{current_calib_id=}"
                                viewpoint.update_calibration(kf_calib[0], kf_calib[1], kf_calib[2])                    
                        

                    else:
                        
                        """
                        Dense Bundle Adjustment (pose, Gaussians)
                        """
                        # self.keyframe_optimizers = torch.optim.Adam(pose_opt_params)
                        # self.keyframe_optimizers.zero_grad()
                        # self.calibration_optimizers = None

                        self.map(self.current_window, iters=iter_per_kf)

                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                    CC = viewpoint.camera_center.cpu().numpy()
                    exposure_a = viewpoint.exposure_a.data.item()
                    exposure_b = viewpoint.exposure_b.data.item()
                    rich.print(f"[bold blue]BackEnd  Optimize:[/bold blue] [{cur_frame_idx:05d}]: fx: {cur_keyframe.fx:.3f}, fy: {cur_keyframe.fy:.3f}, kappa: {cur_keyframe.kappa:.6f}, calib_id: {cur_keyframe.calib_id}. cam_center: ({CC[0]:.3f}, {CC[1]:.3f}, {CC[2]:.3f}), exposure: (a: {exposure_a:.5f}, b: {exposure_b:.5f}). iter_per_kf: {iter_per_kf}\n")

                else:
                    raise Exception("Unprocessed data", data)
        
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()        
        return
