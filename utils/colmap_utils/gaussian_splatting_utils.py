import os
import json
import numpy as np
import open3d as o3d
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

from colmap_utils.colmap import ColMap
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.general_utils import PILtoTorch


# 3DGS colamp_loader
from gaussian_splatting.scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text



# a function to create a list of Camera classes in 3DGS/MonoGS
def assemble_3DGS_cameras(colmap : ColMap, downsample_scale = 1.0):
    camera_stack = []
    camera_centers = []
    posed_image_stack = colmap.getCamPosedImages()

    for image_id, item in colmap.getCamPosedImages().items():
        R, T, imgname, K, kappa = item
        print(f"All cameras: fx = {K[0, 0]:.5f},  fy = {K[1, 1]:.5f},  cx = {K[0, 2]:.5f},  cy = {K[1, 2]:.5f}, kappa = {kappa:.8f}")
    print(f"colmap.avg_K = {colmap.avg_K}, colmap.avg_distort = {colmap.avg_distort}")

    avg_K = colmap.avg_K  / downsample_scale
    kappa = colmap.avg_distort[0]

    fx, fy, cx, cy = avg_K[0, 0], avg_K[1, 1], avg_K[0, 2], avg_K[1, 2]
    print(f"fx = {fx:.5f},  fy = {fy:.5f},  cx = {cx:.5f},  cy = {cy:.5f}, kappa = {kappa:.8f}")

    for image_id, item in posed_image_stack.items():
        R, T, imgname, K_, kappa_ = item
        
        image_path = os.path.join(colmap.image_dir, os.path.basename(imgname))
        image = Image.open(image_path)
        # adjust image resolution if necessary
        orig_w, orig_h = image.size
        imgsize = round(orig_w/(downsample_scale)), round(orig_h/(downsample_scale))

        resized_image_rgb = PILtoTorch(image, imgsize)
        gt_image = resized_image_rgb[:3, ...]

        image_height = gt_image.shape[1]
        image_width = gt_image.shape[2]

        cam = Camera (
                    uid = image_id,
                    color = gt_image,
                    depth = None,
                    image_height = image_height,
                    image_width = image_width,
                    R = R, T = T,
                    fx = fx,
                    fy = fy,
                    cx = cx,
                    cy = cy,
                    fovx = None,
                    fovy = None,
                    kappa = kappa,
                    trans=np.array([0.0, 0.0, 0.0]),
                    scale=1.0,
                    gt_alpha_mask = None,
                    calib_id=1,
                    device="cuda:0",
        )
        camera_stack.append(cam)
        camera_centers.append( - R.transpose() @ T.reshape((3, 1)) ) # camera center
    # getNerfppNorm copied from 3DGS original implementation
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    center, diagonal = get_center_and_diag(camera_centers)
    radius = diagonal * 1.1
    translate = -center
    return camera_stack, {"translate": translate, "radius": radius}





def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    camera_stack = []
    camera_centers = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            principal_point_x = intr.params[2]
            principal_point_y = intr.params[3]

        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # (R, T) : world-to-camera transform
        cam = Camera (
                    uid = uid,
                    color = image,
                    depth = None,
                    image_height = height,
                    image_width = width,
                    R = R, T = T,
                    fx = focal_length_x,
                    fy = focal_length_y,
                    cx = principal_point_x,
                    cy = principal_point_y,
                    fovx = None,
                    fovy = None,
                    kappa = kappa,
                    trans=np.array([0.0, 0.0, 0.0]),
                    scale=1.0,
                    gt_alpha_mask = None,
                    device="cuda:0",
        )
        camera_stack.append(cam)
        camera_centers.append( - R.transpose() @ T.reshape((3, 1)) ) # camera center
    sys.stdout.write('\n')
    return camera_stack








def assemble_3DGS_cameras_from_3DGS_JSON_file (camera_json_file):
    """
    There is no image in the assembled Camera class

    refer to:
        gaussian_splatting/utils/camera_utils.py
        def camera_to_JSON(id, camera : Camera)
        -----------------------------------------------------
                    C2W = np.linalg.inv(Rt)
                    pos = C2W[:3, 3]
                    rot = C2W[:3, :3]

                    camera_entry = {
                        'id' : id,
                        'img_name' : camera.image_name,
                        'width' : camera.width,
                        'height' : camera.height,
                        'position': pos.tolist(),
                        'rotation': serializable_array_2d,
                        'fy' : fov2focal(camera.FovY, camera.height),
                        'fx' : fov2focal(camera.FovX, camera.width)
                    }
    """
    camera_stack = []

    with open(camera_json_file, 'r') as json_file:
        contents = json.load(json_file)

    for cam_info in contents:
        uid = cam_info["id"]
        img_name = cam_info["img_name"]
        W = cam_info["width"]
        H = cam_info["height"]
        pos = np.array ( cam_info["position"] )
        rot = np.array ( cam_info["rotation"] )
        fx = cam_info["fx"]
        fy = cam_info["fy"]

        cx = (W+1)*0.5
        cy = (H+1)*0.5
        kappa = 0.0

        gt_image = None

        # get the world-to-camera transform and set R, T
        W2C_R = np.transpose(rot)
        W2C_T = - np.transpose(rot) @ pos

        cam = Camera (
                    uid = uid,
                    color = gt_image,
                    depth = None,
                    image_height = H,
                    image_width = W,
                    R = W2C_R, T = W2C_T,
                    fx = fx,
                    fy = fy,
                    cx = cx,
                    cy = cy,
                    fovx = None,
                    fovy = None,
                    kappa = kappa,
                    trans=np.array([0.0, 0.0, 0.0]),
                    scale=1.0,
                    gt_alpha_mask = None,
                    calib_id=1,
                    device="cuda:0",
        )
        camera_stack.append(cam)
    return camera_stack




def create_trajectory_lineset(viewpoint_stack, color=[0, 0, 1]):
    camera_centers = []
    for viewpoint in viewpoint_stack:
        camera_centers.append ( viewpoint.camera_center.detach().cpu().numpy() )
    points = np.array( camera_centers )

    lines = []
    for i in range(len(camera_centers)-1):
        lines.append( [i, i+1] )

    colors = [color for i in range(len(lines))]

    odometry_line_set = o3d.geometry.LineSet()
    odometry_line_set.points = o3d.utility.Vector3dVector(points)
    odometry_line_set.lines = o3d.utility.Vector2iVector(lines)
    odometry_line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return odometry_line_set










if __name__ == "__main__":
    

    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)



    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud.  3. Calibrations
    positions, colors = reconstruction.getPointCloud()
    posed_img_stack = reconstruction.getCamPosedImages()

    # interface to 3DGS
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction)

    sparse_depth_stack = reconstruction.getSparseDepthFromImage(image_id = 1)


    try:

        # Create a visualizer
        WIDTH = 1280
        HEIGHT = 720

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=WIDTH, height=HEIGHT)


        # add poinit-cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        # opt.point_show_normal = True
        
        
        # add cameras
        for image_id, item in posed_img_stack.items():
            R, T, imgname, K, kappa = item
            intrinsic = K            
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic, extrinsic=extrinsic)
            vis.add_geometry(cameraLines)

        odometryLines = create_trajectory_lineset(viewpoint_stack)
        vis.add_geometry(odometryLines)


        # visualize and block
        vis.run()
    
    except:
        pass