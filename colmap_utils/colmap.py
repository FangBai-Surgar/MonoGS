
import numpy as np
import pathlib
import pycolmap


import open3d as o3d


# pip install pycolmap
class ColMap:
    """
    PyCOLMAP: APIs
    https://colmap.github.io/pycolmap/index.html
    
    """

    def __init__ (self, image_dir = None):

        self.reconstruction = None

        self.image_dir = image_dir

        self.single_cam_id = 1

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

        '''
        CameraMode
        https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.CameraMode
            AUTO= <CameraMode.AUTO: 0>
            PER_FOLDER= <CameraMode.PER_FOLDER: 2>
            PER_IMAGE= <CameraMode.PER_IMAGE: 3>
            SINGLE= <CameraMode.SINGLE: 1>
        '''
        # print(f"camera_mode = {pycolmap.CameraMode(2)},   also = {pycolmap.CameraMode.PER_FOLDER}")
        '''
        ImageReaderOptions
        https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.ImageReaderOptions
            camera_model
            camera_params
            existing_camera_id: Whether to explicitly use an existing camera for all images. Note that in this case the specified camera model and parameters are ignored. (int, default: -1)
        '''
        # image_reader_options = pycolmap.ImageReaderOptions(existing_camera_id = 1)
        '''
        extract_features
        https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.extract_features
        '''
        pycolmap.extract_features(database_path, image_dir,
                                  camera_mode = pycolmap.CameraMode.SINGLE,
                                  camera_model = 'SIMPLE_RADIAL',
                                  reader_options = pycolmap.ImageReaderOptions(existing_camera_id = 1))
        

        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

        # sparse reconstruction
        self.reconstruction = maps[0]
        # print(self.reconstruction.summary())

        # use single camera intrinsic calibration for all images
        self.__set_to_single_camera()
        '''
        bundle_adjustment
        https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.bundle_adjustment
        '''
        pycolmap.bundle_adjustment(self.reconstruction)
        print(self.reconstruction.summary())

        # save
        self.reconstruction.write(output_path)
        # self.reconstruction.write_text(output_path )  # text format
        self.reconstruction.export_PLY(output_path / "points3D.ply")  # PLY format

        # dense reconstruction
        # pycolmap.undistort_images(mvs_path, output_path, image_dir)
        # pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
        # pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


    def bundleAdjustmentByGivenCalibration (self, focal = None, kappa = None, delta_focal = None):
        # set new camera intrinsics
        self.__set_to_single_camera(focal=focal, kappa=kappa, delta_focal=delta_focal)
        # Bundle Adjustment with fixed camera intrinsics
        '''
        BundleAdjustmentOptions
        https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.BundleAdjustmentOptions
        '''
        ba_opts = pycolmap.BundleAdjustmentOptions(refine_focal_length = False, refine_extra_params = False)
        pycolmap.bundle_adjustment(self.reconstruction, options = ba_opts)
        return self.reconstruction



    def getPoints3DXYZ(self):
        points3d = {}
        for point3D_id, point3D in self.reconstruction.points3D.items():
            points3d[  point3D_id ] = point3D.xyz
            if point3D.track.length() == 1:
                print(f"point3D: id = {point3D_id}. values = {point3D}")
        return points3d


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
        calib_dict, avg_K, avg_kappa = self.__get_calibration()
        posed_image_dict = {}
        for image_id, image in self.reconstruction.images.items():
            pose = image.cam_from_world
            qvec = pose.rotation.quat
            tvec = pose.translation
            # [ R, T ] is a tranformation from world frame to camera frame
            R = self.__qvec2rotmat( qvec )
            T = np.array( tvec )
            (K, kappa) = calib_dict[ image.camera_id ]
            posed_image_dict[image_id] = (R, T, image.name, K, kappa)
        return posed_image_dict



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



    def getSparseKeypointsFromImage (self, image_id,  downsample_scale = 1.0):
        scale_factor = 1.0 / downsample_scale
        image_points = self.reconstruction.images[image_id].points2D
        sparse_keypoints_dict = {}
        for pt in image_points:
            if pt.has_point3D():
                xy_value = np.array( [ pt.xy[0]*scale_factor, pt.xy[1]*scale_factor ] )
                sparse_keypoints_dict[ pt.point3D_id ] =  xy_value
        return sparse_keypoints_dict
    



    def __get_calibration(self):        
        calib_dict = {}
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
                calib_dict[camera_id] = (K, kappa)
                avg_K += K
                avg_kappa += kappa
        return calib_dict,  avg_K/len(calib_dict),  avg_kappa/len(calib_dict)


    @staticmethod
    # copied from 3DGS colmap.loader.py
    def __qvec2rotmat(qvec):
        return np.array([
            [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,   2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],  2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],   1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,  2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],   2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],  1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


    def __set_to_single_camera(self, focal = None, kappa = None, delta_focal = None):        
        # set all cameras to the same camera
        for image_id, image in self.reconstruction.images.items():
            image.camera_id = self.single_cam_id
        
        if focal is not None:
            self.reconstruction.cameras[ self.single_cam_id ].params[0] = focal

        if kappa is not None:
            self.reconstruction.cameras[ self.single_cam_id ].params[3] = kappa

        if delta_focal is not None:
            self.reconstruction.cameras[ self.single_cam_id ].params[0] += delta_focal

        return self.single_cam_id



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
    pose[0:3, 0:3] = R
    pose[0:3, 3] = T
    # print(f"pose = \n {pose}\n")

    img_size = lst[8]
    width, height = int(img_size[0]), int(img_size[1])

    return (K, pose, width, height)




if __name__ == "__main__":

    image_dir = "/home/fang/SURGAR/Colmap_Test/Fountain/images"

    # perform colmap reconstruction
    reconstruction = ColMap(image_dir)



    # extract reconstruction information: 1. posedCameras, 2. 3Dpointcloud.  3. Calibrations
    positions, colors = reconstruction.getPointCloud()
    posed_img_stack = reconstruction.getCamPosedImages()


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


        # visualize and block
        vis.run()
    
    except:
        pass