
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
from slam_impl.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d
import copy


from utils.colmap_utils.colmap import ColMap
from utils.colmap_utils.gaussian_splatting_utils import assemble_3DGS_cameras



import pickle 
import rich

from sfm import SFM, print_viewpoint_stack




from gaussian_viewer import Viewer, create_gaussians_gl


from slam_impl.eval_utils import evaluate_evo, eval_ate, eval_rendering


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


from utils.matplot_utils import annotate_image

# from utils.gtsam_utils.bundle_adjustment import bundle_adjustment




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
    dist = np.array(lst[3])

    R = np.array(lst[4:7])
    T = lst[7]

    pose = np.eye(4)
    pose[0:3, 0:3] = R #.transpose() # debugged using rotation error
    pose[0:3, 3] = T
    # print(f"pose = \n {pose}\n")
    W2C = np.linalg.inv(pose)

    img_size = lst[8]
    width, height = int(img_size[0]), int(img_size[1])

    return (K, dist, W2C, width, height)





def run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = False, downsample_scale = 2**2,
                    phase1_iter = 200, phase3_iter = 500, phase2_DBA_iter = 100, phase2_CaliDBA_iter = 500, phase2_CaliDBA_GSS_iter = 0, use_scale_space = False,
                    set_focal_error = None, save_to_dir = None, COLMAP_same_camera_intrinsics = True):

    # perform colmap reconstruction
    reconstruction = ColMap(image_dir, same_camera_intrinsics=COLMAP_same_camera_intrinsics)

    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction,  downsample_scale = downsample_scale)
    cameras_extent = scale_info["radius"]

    # initialize 3D Gaussians from sparse Colmap output
    gaussians = GaussianModel(sh_degree=0)
    gaussians.spatial_lr_scale = cameras_extent
    
    positions, colors = reconstruction.getPointCloud()
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    gaussians.create_from_pcd(pcd, cameras_extent)
    gaussians.training_setup(opt)

    """
    SFM
    """
    sfm = SFM(pipe, use_gui, copy.deepcopy( viewpoint_stack ), gaussians, opt, cameras_extent)
    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    rich.print("\nPHASE 1")
    print_viewpoint_stack(viewpoint_stack)

    """
    Create noisy intial value for DBA-Calib
    """
    viewpoint_stack_reset = None
    fx_init = reconstruction.avg_K[0, 0]
    if set_focal_error is not None:
        rich.print(f"\nSet Focal Length Error:\n\tdelta_focal = {set_focal_error*fx_init}. \n\tPerform BA to enforce this change.")
        reconstruction.bundleAdjustmentByGivenCalibration(delta_focal=set_focal_error*fx_init)
        viewpoint_stack_reset, _ = assemble_3DGS_cameras(reconstruction,  downsample_scale = downsample_scale)        
        rich.print("\nPHASE 2")
        print_viewpoint_stack(viewpoint_stack_reset)


    sfm.optimize(phase1_iter = phase1_iter,
                 phase3_iter = phase3_iter,
                 phase2_DBA_iter = phase2_DBA_iter,
                 phase2_CaliDBA_iter = phase2_CaliDBA_iter,
                 phase2_CaliDBA_GSS_iter = phase2_CaliDBA_GSS_iter,
                 use_scale_space = use_scale_space,
                #  set_focal_error = set_focal_error*fx_init,
                 viewpoint_stack_reset = copy.deepcopy(viewpoint_stack_reset) if set_focal_error is not None else None
                 )

    (W2C_arr, fx_arr, fy_arr, kappa_arr, rendered_images, captured_images, error_images) = sfm.eval_data()

    psnr_array, ssim_array, lpips_array = eval_rendering_metrics(rendered_images, captured_images)

    if save_to_dir is not None:
        pathlib.Path(save_to_dir).mkdir(parents=True, exist_ok=True)
        for idx in range( len(rendered_images) ):
            psnr = psnr_array[idx]

            psnr_str = " {:.2f} ".format(psnr)
            
            rgb = sfm.tensor2rgb(rendered_images[idx])
            fig, ax, _ = annotate_image(rgb, cmap=None, mytext = psnr_str)
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_rendering'+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
            time.sleep(0.01)

            rgb = sfm.tensor2rgb(captured_images[idx])
            fig, ax, _ = annotate_image(rgb, cmap=None, mytext = psnr_str)
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_original'+'.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
            time.sleep(0.01)

            errormap = error_images[idx].permute(1, 2, 0).contiguous().cpu().numpy()
            fig, ax, im = annotate_image(errormap, cmap='hot', mytext = psnr_str)
            plt.colorbar(im)
            plt.savefig(os.path.join(save_to_dir, str(idx)+'_errormap'+'.png'), bbox_inches='tight', pad_inches=0)            
            plt.close()
            time.sleep(0.01)


    # Fig = Viewer(viewpoint_stack=sfm.viewpoint_stack,  gaussians_gl= create_gaussians_gl(sfm.gaussians))
    uid_arr = []
    for viewpoint in viewpoint_stack:
        uid_arr.append(viewpoint.uid)

    posed_image_dict = reconstruction.getCamPosedImages()
    gt_W2C_dic = {}
    gt_K_dic = {}
    gt_dist_dic = {}
    for image_id, item in posed_image_dict.items():
        uid = image_id
        R, T, imgname, K, kappa = item
        (gt_K, gt_dist, gt_pose, width, height) = read_groundtruth_camera(gt_dir + '/' + imgname + '.camera')
        gt_W2C_dic[uid] = gt_pose
        # print("uid ", uid)
        gt_K_dic[uid] = gt_K
        gt_dist_dic[uid] = gt_dist
    gt_W2C_arr = []
    gt_K_arr = []
    gt_dist_arr = []
    for uid in uid_arr:
        gt_W2C_arr.append ( gt_W2C_dic[uid] )
        gt_K_arr.append( gt_K_dic[uid] )
        gt_dist_arr.append( gt_dist_dic[uid] )
    # print(uid_arr)
    
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


    result ={
        "psnr" : float(np.mean(psnr_array)),
        "ssim" : float(np.mean(ssim_array)),
        "lpips" : float(np.mean(lpips_array)),
        "APE_trans" : ape_stat_trans,
        "APE_rot"   : ape_stat_rot,
        "W2C_arr"  : W2C_arr,
        "gt_W2C_arr" : gt_W2C_arr,
        "fx_arr" : fx_arr,
        "fy_arr" : fy_arr,
        "kappa_arr" : kappa_arr,
        "gt_K_arr" : gt_K_arr,
        "gt_dist_arr" : gt_dist_arr,
        "downsample_scale" : downsample_scale
    }

    return result




def format_results_to_latex_str (results):

    latex_str = []

    gssY_str = "GSS \\cmark"
    gssN_str = "GSS \\xmark"

    latex_str.append(f"\\begin{{tabular}}{{ l | cc | cc | cc | cc | cc }}")
    latex_str.append("     & RFE & ATE & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\")
    latex_str.append(f"    & {gssY_str} & {gssN_str} & {gssY_str} & {gssN_str} & {gssY_str} & {gssN_str} & {gssY_str} & {gssN_str} & {gssY_str} & {gssN_str}  \\\\")
    latex_str.append("\\midrule")
    for datasetname in results:

        # latex_str.append(f"\\multicolumn{{11}}{{c}}{{ {datasetname} }}   \\\\")
        latex_str.append("\\midrule")

        for calib_flag in results[datasetname]:

            if calib_flag == "w/o":
                print(results[datasetname].keys())
                result_gssY = results[datasetname][calib_flag][True] #  w/. COLMAP_GBA
                result_gssN = results[datasetname][calib_flag][False] # w/o COLMAP_GBA
            else:
                result_gssY = results[datasetname][calib_flag][True]
                result_gssN = results[datasetname][calib_flag][False]

            downsample_scale = [ result_gssY["downsample_scale"], result_gssN["downsample_scale"] ]

            gt_fx     = [ result_gssY["gt_K_arr"][-1][0, 0], result_gssN["gt_K_arr"][-1][0, 0] ]
            gt_kappa  = [ result_gssY["gt_dist_arr"][-1][0], result_gssN["gt_dist_arr"][-1][0] ]
            fx        = [ result_gssY["fx_arr"][-1] * downsample_scale[0],  result_gssN["fx_arr"][-1] * downsample_scale[1] ]
            kappa     = [ result_gssY["kappa_arr"][-1],  result_gssN["kappa_arr"][-1] ]

            RCE_focal = [ abs( (fx[0] - gt_fx[0]) / gt_fx[0] ),   abs( (fx[1] - gt_fx[1]) / gt_fx[1] ) ]
            # RCE_kappa = abs( (kappa - gt_kappa) / gt_kappa ) # gt_kappa = 0

            APE_t     = [ result_gssY["APE_trans"], result_gssN["APE_trans"] ]
            APE_r     = [ result_gssY["APE_rot"], result_gssN["APE_rot"] ]

            gs_psnr   = [ result_gssY["psnr"], result_gssN["psnr"] ]
            gs_ssim   = [ result_gssY["ssim"], result_gssN["ssim"] ]
            gs_lpips  = [ result_gssY["lpips"], result_gssN["lpips"] ]

            if calib_flag != "w/o":
                calib_output_str = calib_flag 
                latex_str.append( f"\\#{datasetname} [{calib_output_str}] & {100*RCE_focal[0]:.3f}\\%  &  {100*RCE_focal[1]:.3f}\\%  &  {APE_t[0]:.5f} & {APE_t[1]:.5f} & {gs_psnr[0]:.2f} & {gs_psnr[1]:.2f} & {gs_ssim[0]:.3f} & {gs_ssim[1]:.3f} & {gs_lpips[0]:.4f} & {gs_lpips[1]:.4f}  \\\\" )
            else:

                calib_output_str = "COLMAP+SIFT"
                latex_str.append( f"\\#{datasetname} [{calib_output_str}] & {100*RCE_focal[1]:.3f}\\%  &  {100*RCE_focal[1]:.3f}\\%  &  {APE_t[1]:.5f} & {APE_t[1]:.5f} & {gs_psnr[1]:.2f} & {gs_psnr[1]:.2f} & {gs_ssim[1]:.3f} & {gs_ssim[1]:.3f} & {gs_lpips[1]:.4f} & {gs_lpips[1]:.4f}  \\\\" )

                calib_output_str = "COLMAP+SIFT+GBA"
                latex_str.append( f"\\#{datasetname} [{calib_output_str}] & {100*RCE_focal[0]:.3f}\\%  &  {100*RCE_focal[0]:.3f}\\%  &  {APE_t[0]:.5f} & {APE_t[0]:.5f} & {gs_psnr[0]:.2f} & {gs_psnr[0]:.2f} & {gs_ssim[0]:.3f} & {gs_ssim[0]:.3f} & {gs_lpips[0]:.4f} & {gs_lpips[0]:.4f}  \\\\" )

    latex_str.append(f"\\end{{tabular}}")

    return latex_str



if __name__ == "__main__":

    mp.set_start_method('spawn')

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


    opt.require_calibration = True
    opt.allow_lens_distortion = True


    rich.print("dataset=", dataset.__dict__)
    rich.print("pipe=", pipe.__dict__)
    rich.print("opt=", opt.__dict__)


    # opt.iterations = 1000
    # opt.densification_interval = 30
    # opt.opacity_reset_interval = 200
    # opt.densify_from_iter = 49
    # opt.densify_until_iter = 2000
    # opt.densify_grad_threshold = 0.0002


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

    

    datasets_dict = {
        "Herzjesu" : {
            "name" :      "Herzjesu",
            "image_dir":  "/hdd/sfm/Strecha-Herzjesu/Herzjesu/images",
            "gt_dir":     "/hdd/sfm/Strecha-Herzjesu/Herzjesu/groundtruth"
        },
        "Fountain" : {
            "name" :      "Fountain",
            "image_dir":  "/hdd/sfm/Strecha-Fountain/Fountain/images",
            "gt_dir":     "/hdd/sfm/Strecha-Fountain/Fountain/groundtruth"
        }
    }

    
    result_root_dir = os.path.join(os.getcwd(), "result_sfm")
    pathlib.Path(result_root_dir).mkdir(parents=True, exist_ok=True)

    phase1_iter, phase3_iter = 500, 500 # standard 3DGS rountine, camera not optimized
    phase2_DBA_iter, phase2_CaliDBA_iter = 100, 500 # Gaussian is free to optimize
    phase2_CaliDBA_GSS_iter = 200 # Gaussian is fixed when performing scale space optimization

    
    if True:

        downsample_scale = 2**2

        datasetname = "Fountain"
        image_dir, gt_dir,  = datasets_dict[datasetname]["image_dir"], datasets_dict[datasetname]["gt_dir"], 

        """
        GT: 690
        """
        use_GSS = True
        focal_error=0.2
        result = run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = True, downsample_scale = downsample_scale,
                    phase1_iter = phase1_iter,
                    phase3_iter = phase3_iter,
                    phase2_DBA_iter = phase2_DBA_iter,
                    phase2_CaliDBA_iter = phase2_CaliDBA_iter,
                    phase2_CaliDBA_GSS_iter = phase2_CaliDBA_GSS_iter,
                    use_scale_space = use_GSS,
                    set_focal_error=focal_error,
                    save_to_dir=os.path.join(result_root_dir, "Debug", "withCalib"))
        results = { datasetname : { "w/"+str(focal_error) : {True: result, False: result} }  }
        latex_str = format_results_to_latex_str (results)
        for s in latex_str:
            print(s)

        sys.exit()



    if False:
        """
        save rendered images
        """
        downsample_scale = 2
        for datasetname, dataset in datasets_dict.items():

            image_dir, gt_dir = dataset["image_dir"], dataset["gt_dir"]

            result = run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = False, downsample_scale = downsample_scale,
                        phase1_iter = phase1_iter,
                        phase3_iter = phase3_iter,
                        phase2_DBA_iter = phase2_DBA_iter+phase2_CaliDBA_iter+phase2_CaliDBA_GSS_iter,
                        phase2_CaliDBA_iter = 0,
                        phase2_CaliDBA_GSS_iter = 0,
                        save_to_dir=os.path.join(result_root_dir, datasetname, "without"))
            # rich.print(result)
            
            result = run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = False, downsample_scale = downsample_scale,
                        phase1_iter = phase1_iter,
                        phase3_iter = phase3_iter,
                        phase2_DBA_iter = phase2_DBA_iter,
                        phase2_CaliDBA_iter = phase2_CaliDBA_iter,
                        phase2_CaliDBA_GSS_iter = phase2_CaliDBA_GSS_iter,
                        use_scale_space = True,
                        save_to_dir=os.path.join(result_root_dir, datasetname, "withCalib"))
            # rich.print(result)


    if False:

        downsample_scale = 2
    
        results = {}

        for datasetname, dataset in datasets_dict.items():

            results[datasetname] = {}

            image_dir, gt_dir = dataset["image_dir"], dataset["gt_dir"]

            results[datasetname]['w/o'] = {}

            for COLMAP_same_camera_intrinsics in [True, False]:

                # w/o clibration
                result = run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = False, downsample_scale = downsample_scale,
                            phase1_iter = phase1_iter,
                            phase3_iter = phase3_iter,
                            phase2_DBA_iter = phase2_DBA_iter+phase2_CaliDBA_iter+phase2_CaliDBA_GSS_iter,
                            phase2_CaliDBA_iter = 0,
                            phase2_CaliDBA_GSS_iter = 0,
                            COLMAP_same_camera_intrinsics = COLMAP_same_camera_intrinsics
                            )
                results[datasetname]['w/o'][COLMAP_same_camera_intrinsics] = result

                with open( os.path.join(result_root_dir, 'saved_results.pkl'), 'wb') as f:
                    pickle.dump(results, f)

            # continue
    
            # w/. calibration
            for focal_error in [None, 0, -0.1, 0.1]: # None disables global bundle adjustment in COLMAP
            # for focal_error in [0, -0.1, -0.2, 0.1, 0.2]:
                calib_str = 'w/'+str(focal_error) if focal_error !=0 else 'w/'
                results[datasetname][calib_str] = {}
                """
                with and without Gaussian scale space
                """
                for use_GSS in [True, False]:
                    result = run_colmap_sfm (image_dir, gt_dir, pipe, opt, use_gui = False, downsample_scale = downsample_scale,
                            phase1_iter = phase1_iter,
                            phase3_iter = phase3_iter,
                            phase2_DBA_iter = phase2_DBA_iter,
                            phase2_CaliDBA_iter = phase2_CaliDBA_iter,
                            phase2_CaliDBA_GSS_iter = phase2_CaliDBA_GSS_iter,
                            use_scale_space = use_GSS,
                            set_focal_error= focal_error,
                            COLMAP_same_camera_intrinsics = False
                        )
                    results[datasetname][calib_str][use_GSS] = result
                    with open( os.path.join(result_root_dir, 'saved_results.pkl'), 'wb') as f:
                        pickle.dump(results, f)

    else:

        with open( os.path.join(result_root_dir, 'saved_results.pkl'), 'rb') as f:
            results = pickle.load(f)
        # rich.print("results=", results)


        latex_str = format_results_to_latex_str (results)
        with open( os.path.join(result_root_dir, 'sfm_latex_table.txt'), 'w') as f:
            for s in latex_str:
                f.write(s)
                f.write("\n")

        for s in latex_str:
            print(s)


