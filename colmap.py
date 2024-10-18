
import os
import numpy as np
import pathlib
import pycolmap

from PIL import Image


from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.utils.general_utils import PILtoTorch

import open3d as o3d


# pip install pycolmap
class ColMap:

    def __init__ (self, image_dir = None):

        self.reconstruction = None

        self.image_dir = image_dir

        if image_dir is not None:
            self.run(image_dir)


    def run(self, image_dir=None):

        self.image_dir = image_dir

        if image_dir is None:
            return
        image_dir = pathlib.Path(image_dir)

        output_path =  image_dir.parent / "sparse"
        output_path.mkdir(parents="False", exist_ok="True")
        database_path = output_path / "database.db"

        mvs_path = image_dir.parent / "dense"
        mvs_path.mkdir(parents="False", exist_ok="True")

        pycolmap.extract_features(database_path, image_dir)
        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

        # sparse reconstruction
        self.reconstruction = maps[0]
        self.reconstruction.write(output_path )
        # self.reconstruction.write_text(output_path )  # text format
        self.reconstruction.export_PLY(output_path / "points3D.ply")  # PLY format
        print(self.reconstruction.summary())

        # dense reconstruction
        # pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
        # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


    def getPointCloud(self):
        positions = []
        colors = []
        normals = []
        for point3D_id, point3D in self.reconstruction.points3D.items():
            positions.append(point3D.xyz)
            colors.append(point3D.color / 255.0) # use normalzied colors, which willl be passed to SH
        positions = np.array(positions)
        colors = np.array(colors)
        return positions, colors


    # The reconstructed pose of an image is specified as 
    # the projection from world to the camera coordinate system of an image using
    # a quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ).
    # The coordinates of the projection/camera center are given by -R^t * T
    # The local camera coordinate system of an image is defined in a way that:
    #   * the X axis points to the right,
    #   * the Y axis to the bottom,
    #   * the Z axis to the front as seen from the image.
    # Bring a world point X_world to camera frame
    # X_cam = R * X_world  +  t
    def getCamPosedImages(self):
        posed_image_stack = {}
        for image_id, image in self.reconstruction.images.items():
            pose = image.cam_from_world
            qvec = pose.rotation.quat
            tvec = pose.translation
            # [ R, T ] is a tranformation from world frame to camera frame
            R = self.qvec2rotmat( qvec )
            T = np.array( tvec )
            posed_image_stack[image_id] = (R, T, image.name, image.camera_id)
        return posed_image_stack


    def getCalibration(self):        
        calib_stack = {}
        avg_K = np.zeros((3,3))
        avg_kappa = 0.0
        for camera_id, camera in self.reconstruction.cameras.items():
            if camera.model == pycolmap.CameraModelId.SIMPLE_RADIAL:
                fx = camera.params[0]
                fy = camera.params[0]
                cx = camera.params[1]
                cy = camera.params[2]
                kappa = camera.params[3]
                K = np.array([[fx,  0.0, cx],
                            [0.0, fy,  cy],
                            [0.0, 0.0, 1.0]])
                calib_stack[camera_id] = (K, kappa)
                avg_K += K
                avg_kappa += kappa
        return calib_stack,  avg_K/len(calib_stack),  avg_kappa/len(calib_stack)


    @staticmethod
    # copied from 3DGS colmap.loader.py
    def qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,   2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],  2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],   1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,  2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],   2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],  1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


    def getSparseDepthFromImage (self, image_id,  downsample_scale = 1.0):
        scale_factor = 1.0 / downsample_scale
        points3d = self.reconstruction.points3D
        pose = self.reconstruction.images[image_id].cam_from_world
        image_points = self.reconstruction.images[image_id].points2D
        sparse_depth_stack = []
        for pt in image_points:
            if pt.has_point3D():
                xyz_cam = pose * points3d[ pt.point3D_id  ].xyz
                depth = xyz_cam[2]
                value = np.array( [ pt.xy[0]*scale_factor, pt.xy[1]*scale_factor, depth] )
                sparse_depth_stack.append(value)
        return sparse_depth_stack




# a function to create a list of Camera classes in 3DGS/MonoGS
def assemble_3DGS_cameras(colmap : ColMap, downsample_scale = 1.0,  use_same_calib = True):
    camera_stack = []
    camera_centers = []
    calib_stack, avg_K, avg_kappa = colmap.getCalibration()
    posed_image_stack = colmap.getCamPosedImages()

    for image_id, item in posed_image_stack.items():
        R, T, imgname, camera_id = item
        
        image_path = os.path.join(colmap.image_dir, os.path.basename(imgname))
        image = Image.open(image_path)
        # adjust image resolution if necessary
        orig_w, orig_h = image.size
        imgsize = round(orig_w/(downsample_scale)), round(orig_h/(downsample_scale))

        resized_image_rgb = PILtoTorch(image, imgsize)
        gt_image = resized_image_rgb[:3, ...]

        image_height = gt_image.shape[1]
        image_width = gt_image.shape[2]
        
        if use_same_calib:
            fx = avg_K[0, 0]  / downsample_scale
            fy = avg_K[1, 1]  / downsample_scale
            cx = avg_K[0, 2]  / downsample_scale
            cy = avg_K[1, 2]  / downsample_scale
            kappa = avg_kappa / downsample_scale
        else:
            K, kappa = calib_stack[camera_id]
            fx = K[0, 0]  / downsample_scale
            fy = K[1, 1]  / downsample_scale
            cx = K[0, 2]  / downsample_scale
            cy = K[1, 2]  / downsample_scale
            kappa = kappa / downsample_scale

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
    calib_stack, focal0, kappa0 = reconstruction.getCalibration()

    # interface to 3DGS
    pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    viewpoint_stack, scale_info = assemble_3DGS_cameras(reconstruction)

    # sparse_depth_stack = reconstruction.getSparseDepthFromImage(image_id = 1)


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
            R, T, imgname, camera_id = item
            K, kappa = calib_stack[camera_id]
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