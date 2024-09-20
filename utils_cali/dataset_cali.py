import numpy as np
import os
import glob
import torch
import trimesh
from PIL import Image
import cv2
import OpenEXR
import Imath

from utils.dataset import BaseDataset, TUMDataset, ReplicaDataset, EurocDataset, RealsenseDataset
from gaussian_splatting.utils.graphics_utils import focal2fov

class SimulatedParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.color_paths = []
        self.depth_paths = []
        self.n_img = 0
        self.frames = []
        self.poses = []
        if input_folder[-4:] == '4567':
            self.input_folder = [input_folder[:-4]+'4', input_folder[:-4]+'5', input_folder[:-4]+'6', input_folder[:-4]+'7']
        elif input_folder[-2:] == '67' and input_folder[-4:] != '4567':
            self.input_folder = [input_folder[:-2]+'6', input_folder[:-2]+'7']
        else:
            self.input_folder = [input_folder]
        for input in self.input_folder:
            intrinsic_files = glob.glob(f"{input}/*_Intrinsic Data.txt")
            if len(intrinsic_files) == 0:
                raise ValueError("Intrinsic Data file not found")
            with open(intrinsic_files[0], "r") as f:
                for line in f:
                    if "File Name Prefix" in line:
                        parts = line.split(": ")
                        self.img_start_name = parts[1].strip()
                        break
            # the type of sorted(glob.glob(f"{input}/*" + "_*.png"))
            color_paths= sorted(glob.glob(f"{input}/*" + "_*.png"))
            depth_paths = sorted(glob.glob(f"{input}/*" + "_pos*.exr"))
            self.color_paths.extend(color_paths)
            self.depth_paths.extend(depth_paths)
            self.n_img = len(self.color_paths)
            self.load_poses(f"{input}/" + self.img_start_name + "_Camera Position Data.txt", f"{input}/" + self.img_start_name + "_Camera Quaternion Rotation Data.txt", len(color_paths))

    def load_poses(self, path_position, path_quaternion, n_img):
        
        with open(path_position, "r") as f:
            lines_t = f.readlines()

        with open(path_quaternion, "r") as f:
            lines_q = f.readlines()

        frames = []
        for i in range(n_img):
            line_t = lines_t[i]
            parts = line_t.split()
            t = np.array([float(parts[1][:-1]), float(parts[2][:-1]), float(parts[3])]) #xyz in left handed space
            line_q = lines_q[i]
            parts = line_q.split()           
            q = np.array([float(parts[4]), float(parts[1][:-1]), float(parts[2][:-1]), float(parts[3][:-1])]) # left handed space
            q_right = np.array([q[0], -q[1], q[2], -q[3]]) # wxyz in right handed space
            # t_right = np.array([t[0]/100, -t[1]/100, t[2]/100]) #xyz in right handed space m
            t_right = np.array([t[0]*10, -t[1]*10, t[2]*10]) #xyz in right handed space mm

            T_c_w = trimesh.transformations.quaternion_matrix(q_right)
            T_c_w[:3, 3] = t_right

            self.poses += [T_c_w]
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": (T_c_w).tolist(), # input with camera to world
            }

            frames.append(frame)
        self.frames.extend(frames)

class SimulatedDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        if dataset_path[-4:] == '4567':
            self.input_folder = [dataset_path[:-4]+'4', dataset_path[:-4]+'5', dataset_path[:-4]+'6', dataset_path[:-4]+'7']
        elif dataset_path[-2:] == '67' and dataset_path[-4:] != '4567':
            self.input_folder = [dataset_path[:-2]+'6', dataset_path[:-2]+'7']
        else:
            self.input_folder = [dataset_path]
        self.fx , self.fy, self.cx, self.cy, self.width, self.height, self.fovx, self.fovy, self.K = [], [], [], [], [], [], [], [], []
        self.disorted = False
        self.dist_coeffs = []
        self.map1x, self.map1y = [], []
        self.cali_id = []
        self.img_start_name = []
        for input in self.input_folder:
            index = self.input_folder.index(input)
            intrinsic_files = glob.glob(f"{input}/*_Intrinsic Data.txt")
            print(intrinsic_files)
            if len(intrinsic_files) == 0:
                raise ValueError("Intrinsic Data file not found")
            with open(intrinsic_files[0], "r") as f:
                for line in f:
                    if "Focal Length (mm)" in line:
                        parts = line.split(": ")
                        focal_length = float(parts[1].strip())
                    if "Sensor Type" in line:
                        parts = line.split(": ")
                        sensor_type = float((parts[1].strip()).split()[0])
                    if "Sensor Size (mm)" in line:
                        parts = line.split(": ")
                        sizes = parts[1].strip().split() 
                        sensor_sizes = [float(size.split('=')[1]) for size in sizes if '=' in size]
                        sensor_size = sensor_sizes
                    if "images Resolution (pixel)" in line:
                        parts = line.split(": ")
                        width, height = list(map(int, parts[1].split("*")))
                    if "Total Images" in line:
                        parts = line.split(": ")
                        n_img = int(parts[1].strip())
                    if "File Name Prefix" in line:
                        parts = line.split(": ")
                        self.img_start_name.append(parts[1].strip())

            # write the intrinsic parameters
            for i in range(n_img):
                # input index in self.input_folder:
                self.cali_id.append(index)
                self.width.append(width)
                self.height.append(height)
                self.fx.append(focal_length * self.width[i] / sensor_size[0])
                self.fy.append(focal_length * self.height[i] / sensor_size[1])
                self.cx.append(width / 2)
                self.cy.append(height / 2)
                self.fovx.append(focal2fov(self.fx[i], self.width[i]))
                self.fovy.append(focal2fov(self.fy[i], self.height[i]))
                self.K.append( np.array(
                    [[self.fx[i], 0.0, self.cx[i]], [0.0, self.fy[i], self.cy[i]], [0.0, 0.0, 1.0]]
                ) )

                self.dist_coeffs.append(np.zeros(5))
                map1x, map1y = cv2.initUndistortRectifyMap(
                    self.K[i],
                    self.dist_coeffs[i],
                    np.eye(3),
                    self.K[i],
                    (self.width[i], self.height[i]),
                    cv2.CV_32FC1,
                )
                self.map1x.extend(map1x)
                self.map1y.extend(map1y)
        # self.fx = self.fx[0]
        # self.fy = self.fy[0]
        # self.cx = self.cx[0]
        # self.cy = self.cy[0]
        # self.width = self.width[0]
        # self.height = self.height[0]
        # self.fovx = self.fovx[0]
        # self.fovy = self.fovy[0]
        # self.K = self.K[0]
        # self.dist_coeffs = self.dist_coeffs[0]
        # self.map1x = self.map1x[0]
        # self.map1y = self.map1y[0]
        # depth parameters
        # self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.has_depth = True
        self.depth_scale = 5.0

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        fx = self.fx[idx]
        fy = self.fy[idx]
        cx = self.cx[idx]
        cy = self.cy[idx]
        fovx = self.fovx[idx]
        fovy = self.fovy[idx]
        height = self.height[idx]
        width = self.width[idx]
        cali_id = self.cali_id[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            if depth_path.endswith(".exr"):
                exrfile = OpenEXR.InputFile(depth_path)
                dw = exrfile.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                depth = np.frombuffer(exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                depth = depth.reshape(size[1], size[0]) / self.depth_scale
            else:
                depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        print(f"image: {color_path}, cali_id: {cali_id}")
        return image, depth, pose, fx, fy, cx, cy, fovx, fovy, height, width, cali_id
        # return image, depth, pose

class SimulatesDatasets(SimulatedDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = SimulatedParser(dataset_path, config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.focal_changed = True

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        dataset = TUMDataset(args, path, config)
        dataset.focal_changed = False # Dynamically Adding the Attribute
        return dataset
    elif config["Dataset"]["type"] == "replica":
        dataset = ReplicaDataset(args, path, config)
        dataset.focal_changed = False
        return dataset
    elif config["Dataset"]["type"] == "euroc":
        dataset = EurocDataset(args, path, config)
        dataset.focal_changed = False
        return dataset
    elif config["Dataset"]["type"] == "realsense":
        dataset = RealsenseDataset(args, path, config)
        dataset.focal_changed = False
        return dataset
    elif config["Dataset"]["type"] == "simulated":
        return SimulatesDatasets(args, path, config)
    else:
        raise ValueError("Unknown dataset type")