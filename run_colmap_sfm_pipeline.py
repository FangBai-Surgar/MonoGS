
import sys, os

import torch
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gui import gui_utils, sfm_gui, slam_gui

import time
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_splatting.utils.general_utils import safe_state
from utils.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d
import copy


from colmap_utils.colmap import ColMap
from colmap_utils.gaussian_splatting_utils import assemble_3DGS_cameras



import pickle 

from sfm import SFM




from gaussian_viewer import Viewer, create_gaussians_gl


from utils.eval_utils import evaluate_evo, eval_ate, eval_rendering


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

import pathlib
import cv2


from matplot_utils import image_annotation

from gtsam_utils.bundle_adjustment import bundle_adjustment




def eval_rendering_metrics(rendered_images, captured_images):
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        )
    for idx in range(len(rendered_images)):
        image = rendered_images[idx]
        gt_image = captured_images[idx]

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
    return psnr_array, ssim_array, lpips_array


def eval_pose_metrics_translation(poses_gt, poses_est, monocular=True):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=monocular)
    # below old method does not work anymore
    # traj_est_aligned = trajectory.align_trajectory(
    #     traj_est, traj_ref, correct_scale=monocular
    # )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()

    return (ape_stat, ape_stats)



def eval_pose_metrics_rotation(poses_gt, poses_est, monocular=True):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=monocular)
    # below old method does not work anymore
    # traj_est_aligned = trajectory.align_trajectory(
    #     traj_est, traj_ref, correct_scale=monocular
    # )

    ## RMSE
    pose_relation = metrics.PoseRelation.rotation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()

    return (ape_stat, ape_stats)


def read_groundtruth_camera(ground_truth_camera_file):    
    with open(ground_truth_camera_file, 'r') as f:
        lines = f.readlines()
    lst = []
    for line in lines:
        arr = np.fromstring(line, sep=' ')
        lst.append(arr)

    K = np.array(lst[0:3])
    # print(f"K = \n{K}")

    R = np.array(lst[4:7])
    T = lst[7]

    pose = np.eye(4)
    pose[0:3, 0:3] = R #.transpose() # debugged using rotation error
    pose[0:3, 3] = T
    # print(f"pose = \n {pose}\n")
    W2C = np.linalg.inv(pose)

    img_size = lst[8]
    width, height = int(img_size[0]), int(img_size[1])

    return (K, W2C, width, height)





def main(image_dir, gt_dir, downsample_scale = 2**2, phase1_iter = 200, phase3_iter = 500, phase2_DBA_iter = 100, phase2_CaliDBA_iter = 500, phase2_CaliDBA_GSS_iter = 0, set_focal_error = None, save_to_dir = None):

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


    opt.iterations = 100
    opt.densification_interval = 30
    opt.opacity_reset_interval = 200
    opt.densify_from_iter = 49
    opt.densify_until_iter = 2000
    opt.densify_grad_threshold = 0.0002


    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)

    if set_focal_error is not None:
        print(f"\nSet Focal Length Error:\n\tdelta_focal = {set_focal_error}. \n\tPerform BA to enforce this change.")
        # print(f"self.reconstruction.images  = \n{reconstruction.reconstruction.images}")
        # print(f"self.reconstruction.cameras = \n{reconstruction.reconstruction.cameras}")
        reconstruction.bundleAdjustmentByGivenCalibration(delta_focal=set_focal_error)
        # print(f"self.reconstruction.images  = \n{reconstruction.reconstruction.images}")
        # print(f"self.reconstruction.cameras = \n{reconstruction.reconstruction.cameras}")


    if False: # perform bundle adjustment using gtsam
        points3D_dict = reconstruction.getPoints3DXYZ()
        sparse_keypoints_meausurements_dict = {}
        poses_dict = {}
        posed_image_dict = reconstruction.getCamPosedImages()
        for image_id, item in posed_image_dict.items():
            uid = image_id
            R, T, imgname, K, kappa = item
            sparse_keypoints_dict = reconstruction.getSparseKeypointsFromImage (image_id,  downsample_scale = downsample_scale)
            sparse_keypoints_meausurements_dict[ image_id  ] = sparse_keypoints_dict
            W2C = np.eye(4)
            W2C[:3, :3] = R
            W2C[:3, 3] = T
            poses_dict[ image_id ] = W2C
        avg_K = K  / downsample_scale
        avg_K[2, 2] = 1.0
        # perform BA with given calibration K
        bundle_adjustment(kpt_measurements=sparse_keypoints_meausurements_dict,
                        poses_w2c=poses_dict,
                        points=points3D_dict,
                        K=avg_K,
                        compute_marginals=False, plot_figure=True)

        sys.exit()




    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud
    
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction,  downsample_scale = downsample_scale,  use_same_calib = True)
    

    print(f"scale_info = {scale_info}")
    cameras_extent = scale_info["radius"]

    # initialize 3D Gaussians from sparse Colmap output
    gaussians = GaussianModel(sh_degree=0)
    gaussians.spatial_lr_scale = cameras_extent
    
    positions, colors = reconstruction.getPointCloud()
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    gaussians.create_from_pcd(pcd, cameras_extent)
    gaussians.training_setup(opt)

    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    ## visualization
    use_gui = True
    sfm = SFM(pipe, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)
    sfm.require_calibration = True
    sfm.allow_lens_distortion = True

    sfm.optimize(phase1_iter = phase1_iter,
                 phase3_iter = phase3_iter,
                 phase2_DBA_iter = phase2_DBA_iter,
                 phase2_CaliDBA_iter = phase2_CaliDBA_iter,
                 phase2_CaliDBA_GSS_iter = phase2_CaliDBA_GSS_iter)

    (W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images) = sfm.eval_data()


    # Fig = Viewer(viewpoint_stack=sfm.viewpoint_stack,  gaussians_gl= create_gaussians_gl(sfm.gaussians))
    uid_arr = []
    for viewpoint in viewpoint_stack:
        uid_arr.append(viewpoint.uid)


    posed_image_dict = reconstruction.getCamPosedImages()
    gt_W2C_dic = {}
    for image_id, item in posed_image_dict.items():
        uid = image_id
        R, T, imgname, K, kappa = item
        (K, pose, width, height) = read_groundtruth_camera(gt_dir + '/' + imgname + '.camera')
        gt_W2C_dic[uid] = pose
        # print("uid ", uid)
    gt_W2C_arr = []
    for uid in uid_arr:
        gt_W2C_arr.append ( gt_W2C_dic[uid] )
    # print(uid_arr)
    
    psnr_array, ssim_array, lpips_array = eval_rendering_metrics(rendered_images, captured_images)

    gt_C2W_arr, C2W_arr = [], []
    gt_centers, centers = [], []
    for pose in gt_W2C_arr:
        gt_C2W_arr.append( np.linalg.inv(pose) )
        gt_centers.append( np.linalg.inv(pose)[0:3, 3] )
    for pose in W2C_arr:
        C2W_arr.append( np.linalg.inv(pose) )
        centers.append( np.linalg.inv(pose)[0:3, 3] )

    (ape_stat_trans, ape_stats_trans) = eval_pose_metrics_translation(gt_C2W_arr, C2W_arr, monocular=True)
    (ape_stat_rot, ape_stats_rot) = eval_pose_metrics_rotation(gt_C2W_arr, C2W_arr, monocular=True)


    if False: # used to test trajectory alignment
        traj_ref = PosePath3D(poses_se3=gt_C2W_arr)
        traj_est = PosePath3D(poses_se3=C2W_arr)
        traj_est_aligned = copy.deepcopy(traj_est)
        traj_est_aligned.align(traj_ref, correct_scale=True)

        # print(traj_ref.poses_se3)
        # print(traj_est_aligned.poses_se3)
        gt_centers, centers = [], []
        for pose in traj_ref.poses_se3:
            gt_centers.append( pose[0:3, 3] )
        for pose in traj_est_aligned.poses_se3:
            centers.append( pose[0:3, 3] )

        gt_centers = np.array(gt_centers).transpose()
        centers = np.array(centers).transpose()
        ax = plt.axes(projection='3d')
        ax.plot3D(gt_centers[0], gt_centers[1], gt_centers[2], 'red')
        ax.plot3D(centers[0], centers[1], centers[2], 'blue')
        plt.axis('equal')
        plt.show()

    if save_to_dir is not None:
        pathlib.Path(save_to_dir).mkdir(parents=True, exist_ok=True)
        for idx in range( len(rendered_images) ):
            psnr = psnr_array[idx]

            psnr_str = " {:.2f} ".format(psnr)
            
            rgb = sfm.tensor2rgb(rendered_images[idx])
            fig, ax = image_annotation(rgb, cmap=None, mytext = psnr_str)
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_rendering'+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
            time.sleep(0.01)

            rgb = sfm.tensor2rgb(captured_images[idx])
            fig, ax = image_annotation(rgb, cmap=None, mytext = psnr_str)
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_original'+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
            time.sleep(0.01)

            errormap = error_images[idx].permute(1, 2, 0).contiguous().cpu().numpy()
            fig, ax = image_annotation(errormap, cmap='hot', mytext = psnr_str)
            plt.colorbar()
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_errormap'+'.png'), bbox_inches='tight', pad_inches=0)            
            plt.close()
            time.sleep(0.01)



    psnr_mean = float(np.mean(psnr_array))
    ssim_mean = float(np.mean(ssim_array))
    lpips_mean = float(np.mean(lpips_array))

    fx = viewpoint_stack[-1].fx
    fy = viewpoint_stack[-1].fy
    kappa = viewpoint_stack[-1].kappa

    return (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)



if __name__ == "__main__":

    mp.set_start_method('spawn')

    """ DATASET URL
    
    https://colmap.github.io/datasets.html#datasets
    
    https://cvg-data.inf.ethz.ch/

    """

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/South-Building.zip"
    '''
    ground_truth (not provided)
    128 images of the “South” building at UNC Chapel Hill. The images are taken with the same camera, kindly provided by Christopher Zach.
    '''

    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Herzjesu.zip"
    image_dir = "/hdd/sfm/Strecha-Herzjesu/Herzjesu/images"
    gt_dir =    "/hdd/sfm/Strecha-Herzjesu/Herzjesu/groundtruth"
    '''
    ground_truth calibration:
        2759.48 0 1520.69
        0 2764.16 1006.81
    '''
    
    data_url = "https://cvg-data.inf.ethz.ch/local-feature-evaluation-schoenberger2017/Strecha-Fountain.zip"
    image_dir = "/hdd/sfm/Strecha-Fountain/Fountain/images"
    gt_dir =    "/hdd/sfm/Strecha-Fountain/Fountain/groundtruth"
    '''
    ground_truth calibration:
        2759.48 0 1520.69
        0 2764.16 1006.81
    '''

    results = {}

    runSfMDebug = 0
    runBatchExp = 1
    runSaveRendering = 1

    GSS_iter = 0


    if runSfMDebug:

        image_dir = "/hdd/sfm/Strecha-Fountain/Fountain/images"
        gt_dir =    "/hdd/sfm/Strecha-Fountain/Fountain/groundtruth"
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 100,
                                                                                    phase3_iter = 5,
                                                                                    phase2_DBA_iter = 20,
                                                                                    phase2_CaliDBA_iter = 6, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=100,
                                                                                    save_to_dir=os.path.join(os.getcwd(), "Debug/withCalib"))
        print(f"\npsnr = {np.mean(psnr_mean)}\nssim_array = {np.mean(ssim_mean)}\nlpips_array={lpips_mean}\nape_trans={ape_stat_trans}\nape_rot={ape_stat_rot}")
        print(f"fx = {fx}, fy = {fy}, kappa = {kappa}")





    if runSaveRendering:
        image_dir = "/hdd/sfm/Strecha-Herzjesu/Herzjesu/images"
        gt_dir =    "/hdd/sfm/Strecha-Herzjesu/Herzjesu/groundtruth"
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 600+GSS_iter,
                                                                                    phase2_CaliDBA_iter = 0,
                                                                                    phase2_CaliDBA_GSS_iter = 0,
                                                                                    save_to_dir=os.path.join(os.getcwd(), "Herzjesu/without"))
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500,
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    save_to_dir=os.path.join(os.getcwd(), "Herzjesu/withCalib"))
        image_dir = "/hdd/sfm/Strecha-Fountain/Fountain/images"
        gt_dir =    "/hdd/sfm/Strecha-Fountain/Fountain/groundtruth"
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 600+GSS_iter,
                                                                                    phase2_CaliDBA_iter = 0,
                                                                                    phase2_CaliDBA_GSS_iter = 0,
                                                                                    save_to_dir=os.path.join(os.getcwd(), "Fountain/without"))
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500,
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    save_to_dir=os.path.join(os.getcwd(), "Fountain/withCalib"))



    if runBatchExp:
        image_dir = "/hdd/sfm/Strecha-Herzjesu/Herzjesu/images"
        gt_dir =    "/hdd/sfm/Strecha-Herzjesu/Herzjesu/groundtruth"

        # w/o clibration
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 600+GSS_iter,
                                                                                    phase2_CaliDBA_iter = 0,
                                                                                    phase2_CaliDBA_GSS_iter = 0)
        print(f"Herzjesu[w/o]: {(psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)}")
        results["Herzjesu[w/o]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)
    
        # w/ calibration
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500,
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter)
        print(f"Herzjesu[w/.]: {(psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)}")
        results["Herzjesu[w/.]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 50
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=50)
        results["Herzjesu[w/50]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 100
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=100)
        results["Herzjesu[w/100]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 150
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=150)
        results["Herzjesu[w/150]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. -50
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-50)
        results["Herzjesu[w/-50]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. -100
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-100)
        results["Herzjesu[w/-100]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)



        # w/ calibration. -150
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-150)
        results["Herzjesu[w/-150]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)



    if runBatchExp:
        image_dir = "/hdd/sfm/Strecha-Fountain/Fountain/images"
        gt_dir =    "/hdd/sfm/Strecha-Fountain/Fountain/groundtruth"

        # w/o clibration
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 600+GSS_iter,
                                                                                    phase2_CaliDBA_iter = 0,
                                                                                    phase2_CaliDBA_GSS_iter = 0)
        print(f"Fountain[w/o]: {(psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)}")
        results["Fountain[w/o]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)
    
        # w/ calibration
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500,
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter)
        print(f"Fountain[w/.]: {(psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)}")
        results["Fountain[w/.]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 50
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=50)
        results["Fountain[w/50]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 100
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=100)
        results["Fountain[w/100]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. 150
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=150)
        results["Fountain[w/150]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. -50
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-50)
        results["Fountain[w/-50]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. -100
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-100)
        results["Fountain[w/-100]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)


        # w/ calibration. -150
        (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa) = main(image_dir, gt_dir, downsample_scale = 2**2,
                                                                                    phase1_iter = 200,
                                                                                    phase3_iter = 500,
                                                                                    phase2_DBA_iter = 100,
                                                                                    phase2_CaliDBA_iter = 500, 
                                                                                    phase2_CaliDBA_GSS_iter = GSS_iter,
                                                                                    set_focal_error=-150)
        results["Fountain[w/-150]"] = (psnr_mean, ssim_mean, lpips_mean, ape_stat_trans, ape_stat_rot, fx, fy, kappa)




    if runBatchExp:

        print("results")
        print(results)


        with open('saved_results.pkl', 'wb') as f:
            pickle.dump(results, f)


        with open('saved_results.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)

        print("loaded dict")
        print(loaded_dict)

