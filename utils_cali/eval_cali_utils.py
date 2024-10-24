import json
import os
import matplotlib
matplotlib.use('Agg')
import cv2
import evo
import evo.main_config
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log

import copy
import pickle
# write a dict

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    # traj_est_aligned = copy.deepcopy(traj_est)
    # traj_est_aligned.align(traj_ref, correct_scale=monocular)
    # below old method does not work anymore
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.pdf".format(str(label))), dpi=90)

    # if label == "final":
    plot_mode = evo.tools.plot.PlotMode.xyz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    # SETTINGS.plot_axis_marker_scale = 0.1
    # SETTINGS.plot_reference_axis_marker_scale = 0.1
    # SETTINGS.plot_pose_correspondences = True
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_3dplot_{}.pdf".format(str(label))), dpi=90)
    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, *_ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians_class(save_dir, gaussians):
    os.makedirs(os.path.join(save_dir, 'gs'), exist_ok=True)
    with open(save_dir + '/gs/instance.pkl', 'wb') as f:
        pickle.dump(gaussians, f)

def save_cali(save_dir, frames, kf_indices, N_frames=None):
    cali_data = dict()
    cali_id, focal_est, focal_gt = [], [], []
    kappa_est, kappa_gt = [], []
    focal_percentage = []

    for kf_id in kf_indices:
        kf = frames[kf_id]

        cali_id.append(frames[kf_id].uid)
        focal_est.append(frames[kf_id].fx)
        focal_gt.append(frames[kf_id].fx_init)
        focal_percentage.append( abs(frames[kf_id].fx_init - frames[kf_id].fx) / frames[kf_id].fx_init)

        kappa_est.append(frames[kf_id].kappa)
        kappa_gt.append(frames[kf_id].kappa_init)

    cali_data["cali_id"] = cali_id
    cali_data["focal_est"] = focal_est
    cali_data["focal_gt"] = focal_gt
    cali_data["kappa_est"] = kappa_est
    cali_data["kappa_gt"] = kappa_gt
    cali_data["focal_percentage"] = focal_percentage

    cali_dir = os.path.join(save_dir, "cali")
    plot_dir = os.path.join(save_dir, "cali", "plot")
    mkdir_p(cali_dir)
    mkdir_p(plot_dir)
    json.dump(
        cali_data,
        open(os.path.join(cali_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    plt.figure(figsize=(10, 6))
    plt.plot(cali_data['cali_id'], cali_data['focal_percentage'], marker='o')
    plt.title('Focal Percentage vs Frame ID')
    plt.xlabel('Frame ID')
    plt.ylabel('Focal Percentage')
    plt.grid(True)

    # Save the plot in the plot directory
    plot_file_path_pdf = os.path.join(plot_dir, 'focal_percentage_vs_cali_id.pdf')
    plt.savefig(plot_file_path_pdf)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(cali_data['cali_id'], cali_data["focal_gt"], marker='o', label='Focal Ground Truth')
    plt.plot(cali_data['cali_id'], cali_data["focal_est"], marker='o', label='Focal Estimate')
    plt.title('Focal vs Frame ID')
    plt.xlabel('Frame ID')
    plt.ylabel('Focal')
    plt.grid(True)
    # Display the legend
    plt.legend()

    # Save the plot in the plot directory
    plot_file_path_pdf = os.path.join(plot_dir, 'focal_vs_cali_id.pdf')
    plt.savefig(plot_file_path_pdf)
    plt.close()
