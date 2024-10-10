
import sys, os

import torch
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.gaussian_model_GS import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gui import gui_utils, sfm_gui

import time
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from gaussian_splatting.utils.general_utils import safe_state
from utils.multiprocessing_utils import FakeQueue, clone_obj

from PIL import Image
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d



from sfm import SFM
from depth_anything import DepthAnything
from colmap import ColMap
from colmap import assemble_3DGS_cameras



def init_dense_pcd_from_network (viewpoint_stack, reconstruction: ColMap, num_points = 20000):

    pcd_downsample_factor = viewpoint_stack[0].image_height * viewpoint_stack[0].image_width * len(viewpoint_stack) / num_points

    DA = DepthAnything()

    positions = None
    colors = None

    for cam in viewpoint_stack:

        sparse_depth_stack = reconstruction.getSparseDepthFromImage(image_id = cam.uid, downsample_scale = downsample_scale )
        rgb_raw = (cam.original_image *255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        # use depth prediction from a Neural network
        disp_raw = DA.eval(rgb_raw)
        depth_raw = 10.0 / disp_raw  # depth = (focal * baseline) / disparity

        # depth_rect = DA.correct_depth_from_sparse_points (depth=depth_raw, uv_depth_stack=sparse_depth_stack)

        scale = DA.estimateScaleFactor(depth=depth_raw, uv_depth_stack=sparse_depth_stack)
        depth_rect = depth_raw * scale
        print(f"depth scale correction = {scale}, rgb_raw.shape = {rgb_raw.shape} depth_raw.shape = {depth_raw.shape}, depth_rect.shape = {depth_rect.shape}")


        if False:
            plt.rcParams["figure.figsize"] = (15, 6)
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            ax1.imshow(rgb_raw)
            ax2.imshow(depth_raw)
            ax3.imshow(depth_rect)
            plt.show()


        # RGB-D image to pcd in world frame
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depth_rect.astype(np.float32))
        new_xyz, new_rgb = GaussianModel.create_pcd_from_image_and_depth(cam, rgb, depth, downsample_factor = pcd_downsample_factor)
        
        positions = np.concatenate((positions, new_xyz), axis=0) if positions is not None else new_xyz
        colors = np.concatenate((colors, new_rgb), axis=0) if colors is not None else new_rgb

    return positions, colors




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


    opt.iterations = 5000
    opt.densification_interval = 50
    opt.opacity_reset_interval = 350
    opt.densify_from_iter = 49
    opt.densify_until_iter = 4000
    opt.densify_grad_threshold = 0.0002







    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"
    '''
    ground_truth calibration:
        2759.48 0 1520.69
        0 2764.16 1006.81
    '''
    

    use_pcd_from_depth_prediction = True


    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)

    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud
    downsample_scale = 2**2
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


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    ## visualization
    use_gui = True
    q_main2vis = mp.Queue() if use_gui else FakeQueue()
    q_vis2main = mp.Queue() if use_gui else FakeQueue()

    if use_gui:
        bg_color = [0.0, 0.0, 0.0]
        params_gui = gui_utils.ParamsGUI(
            pipe=pipe,
            background=torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            gaussians=GaussianModel(dataset.sh_degree),
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        gui_process = mp.Process(target=sfm_gui.run, args=(params_gui,))
        gui_process.start()
        time.sleep(3)

    print(f"Run with image W: { viewpoint_stack[0].image_width },  H: { viewpoint_stack[0].image_height }")


    sfm = SFM(pipe, q_main2vis, q_vis2main, use_gui, viewpoint_stack, gaussians, opt, cameras_extent)

    # From dense depth prediction of a neural network
    if use_pcd_from_depth_prediction:
        positions, colors = init_dense_pcd_from_network(viewpoint_stack, reconstruction, num_points = 100000)
        sfm.add_dense_point_cloud(positions=positions, colors=colors)


    sfm.MODULE_TEST_CALIBRATION = False

    

    sfm.start_calib_iter = 1300
    sfm.stop_calib_iter = 500

    sfm.start_pose_iter = 1100
    sfm.stop_pose_iter = 500

    sfm.start_gaussian_iter = 0
    sfm.stop_gaussian_iter = 100000

    sfm.add_dense_pcd_iter = max(sfm.stop_calib_iter, sfm.stop_pose_iter) + 1


    sfm.require_calibration = True
    sfm.allow_lens_distortion = True
    

    sfm_process = mp.Process(target=sfm.optimize)
    sfm_process.start()

  
    torch.cuda.synchronize()


    if use_gui:
        gui_process.join()
        sfm_gui.Log("GUI Stopped and joined the main thread", tag="GUI")


    sfm_process.join()
    sfm_gui.Log("Finished", tag="SfM")


