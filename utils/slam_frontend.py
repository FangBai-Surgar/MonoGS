import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth

from optimizers import CalibrationOptimizer, PoseOptimizer, lr_exp_decay_helper

from gaussian_scale_space import image_conv_gaussian_separable
import rich


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
        self.MODULE_TEST_CALIBRATION = False
        self.signal_calibration_change = False


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

    def tracking(self, cur_frame_idx, viewpoint, continue_optimize=False):
        if (not continue_optimize):
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.update_RT(prev.R, prev.T)
        
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
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
        for tracking_itr in range(self.tracking_itr_num):
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
            )
            loss_tracking.backward()

            with torch.no_grad():
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

        for kf_id, kf_R, kf_T, kf_fx, kf_fy, kf_kappa in keyframes:
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
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
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
                    if cur_frame_idx < 100:
                        viewpoint.calibration_identifier = 0
                        focal_ref = None
                    elif cur_frame_idx >= 100 and cur_frame_idx < 200:
                        viewpoint.calibration_identifier = 1
                        focal_ref = 400
                    elif cur_frame_idx >= 200 and cur_frame_idx < 300:
                        viewpoint.calibration_identifier = 2
                        focal_ref = 350
                    elif cur_frame_idx >= 300 and cur_frame_idx < 400:
                        viewpoint.calibration_identifier = 3
                        focal_ref = 700
                    elif cur_frame_idx >= 400 and cur_frame_idx < 500:
                        viewpoint.calibration_identifier = 4
                        focal_ref = 900
                    else:
                        viewpoint.calibration_identifier = 4
                        focal_ref = None
                        
                
                if len(self.cameras) > self.use_every_n_frames:
                    prev = self.cameras[cur_frame_idx - self.use_every_n_frames] # last frame in tracking
                    viewpoint.update_calibration (prev.fx, prev.fy, prev.kappa) # use last frame calibration
                    viewpoint.update_RT(prev.R, prev.T) # use last frame pose
                    if viewpoint.calibration_identifier != prev.calibration_identifier:
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
                    lr = self.init_focal (viewpoint, optimizer_type = "Adam", gaussian_scale_t = 10.0,  beta = 0.0, learning_rate = 0.1, max_iter_num = 30, step_safe_guard = False)
                    self.init_focal (viewpoint, optimizer_type = "SGD", gaussian_scale_t = 0.0,  beta = 1.0, learning_rate = lr, max_iter_num = 20, step_safe_guard = True)

                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                if self.require_calibration and self.initialized and self.signal_calibration_change:
                    self.init_focal (viewpoint, optimizer_type = "SGD", gaussian_scale_t = 0.0,  beta = 0.0, learning_rate = lr, max_iter_num = 20, step_safe_guard = True)
                    render_pkg = self.tracking(cur_frame_idx, viewpoint, continue_optimize=True) # render again with the best parameters
    

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
                    rich.print(f"[bold blue]FrontEnd Send    :[/bold blue] [{cur_frame_idx}]: fx: {viewpoint.fx:.3f}, fy: {viewpoint.fy:.3f}, kappa: {viewpoint.kappa:.6f}, calib_id: {viewpoint.calibration_identifier}")
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



    
    def init_focal (self, viewpoint, optimizer_type = "Adam", gaussian_scale_t = 5.0, beta = 1.0, learning_rate = 0.1, max_iter_num = 20, step_safe_guard = False):

        viewpoint_stack = []
        viewpoint_stack.append(viewpoint)

        H = viewpoint.image_height
        W = viewpoint.image_width
        focal_ref = np.sqrt(H*H + W*W)/2

        calibration_optimizers = CalibrationOptimizer(viewpoint_stack, focal_ref, focal_optimizer_type= optimizer_type) # only one view
        calibration_optimizers.num_line_elements = max_iter_num # sample points for line fitting
        rich.print(f"[bold green]Initialize focal length optimizer: {optimizer_type}, lr = {learning_rate} [/bold green]")
        calibration_optimizers.update_focal_learning_rate(lr = learning_rate)

        rgb_boundary_threshold = 0.01

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

            gt_image = viewpoint.original_image.cuda() 
            mask = (gt_image.sum(dim=0) > rgb_boundary_threshold)

            # loss function
            if gaussian_scale_t > 0.5: # Gaussian scale space 
                image_scale_t = image_conv_gaussian_separable(image, sigma=gaussian_scale_t, epsilon=0.01)
                gt_image_scale_t = image_conv_gaussian_separable(gt_image, sigma=gaussian_scale_t, epsilon=0.01)
            else:
                image_scale_t = image
                gt_image_scale_t = gt_image

            
            huber_loss_function = torch.nn.SmoothL1Loss(reduction = 'mean', beta = beta) # beta = 0, this becomes l1 loss
            loss = huber_loss_function(image_scale_t*mask, gt_image_scale_t*mask)
            
            # print(f"focal_init: iter: [{itr}]")
            if step_safe_guard and (loss > loss_prev):
                rich.print(f"[bold yellow][Warning]: learning rate is too big! revoke previous step and shrink learning rate[/bold yellow]")
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
            last_keyframe = self.cameras[last_keyframe_idx] # last keyframe (optimzied by backend)
            last_frame = self.cameras[cur_frame_idx - self.use_every_n_frames] # last frame in tracking
            if (last_keyframe.calibration_identifier == last_frame.calibration_identifier):
                last_frame.update_calibration (last_keyframe.fx, last_keyframe.fy, last_keyframe.kappa)


    


