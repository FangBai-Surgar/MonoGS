
import os
import numpy as np
import glob


from slam_impl.camera_utils import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.system_utils import mkdir_p

from slam_impl.dataset import load_dataset, ReplicaParser, TUMParser, EuRoCParser, RealsenseDataset


import yaml
from munch import munchify
from slam_impl.config_utils import load_config

import json
from PIL import Image

import matplotlib.pyplot as plt
import torch
import rich

try:
    import pyrealsense2 as rs
except Exception:
    pass



class SLAM_Dataset_Simulator:

    def __init__(self, config_file, result_path):

        self.config_file = config_file

        self.config = load_config(config_file)        
        self.ply_path = os.path.join(result_path, "point_cloud/final/point_cloud.ply")
        self.cameras_path = os.path.join(result_path, "cameras/iteration_final.json")


        model_params = munchify(self.config ["model_params"])
        opt_params = munchify(self.config ["opt_params"])
        pipeline_params = munchify(self.config ["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        """
        read 3D gaussians
        """
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.load_ply(self.ply_path)


        """
        read ground-truth poses
        """
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=self.config
        )
        self.__correct_dataset_groundtruth_poses(cameras_path = self.cameras_path )

        self.poses = self.dataset.poses
        self.color_paths = self.dataset.color_paths
        self.depth_paths = self.dataset.depth_paths

        """
        initial calibration
        """
        calibration = self.config["Dataset"]["Calibration"]
        self.fx_init = calibration["fx"]
        self.fy_init = calibration["fy"]
        self.cx_init = calibration["cx"]
        self.cy_init = calibration["cy"]
        self.kappa_init = 0
        self.aspect_ratio = self.fy_init / self.fx_init

        self.fx_arr = self.fx_init * np.ones(len(self.poses))
        self.fy_arr = self.fy_init * np.ones(len(self.poses))
        self.kappa_arr = self.kappa_init * np.ones(len(self.poses))


        self.idx_update = []
        self.fx_update = []
        self.fy_update = []
        self.kappa_update = []

        print(len(self.dataset.color_paths))
        print(len(self.dataset.depth_paths))
        print(len(self.dataset.poses))

        """
        depth
        """
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None



    def __correct_dataset_groundtruth_poses (self, cameras_path):
        def load_cameras(file_path):
            try:
                with open(file_path, 'r') as file:
                    return json.load(file)
            except FileNotFoundError:
                print(f"File not found: {file_path}")

        poses_json = load_cameras(cameras_path)
        R_stack = poses_json["R"]
        T_stack = poses_json["T"]

        est_poses = []
        for R, T in zip(R_stack, T_stack):
            pose = np.eye(4)
            pose[0:3, 0:3] = np.array(R)
            pose[0:3, 3] = np.array(T)
            est_poses.append( pose )

        self.dataset.poses = est_poses.copy()



    def __render_impl(self, cur_frame_idx, fx=None, fy=None, kappa=None):
        viewpoint = Camera.init_from_dataset(
                            self.dataset, cur_frame_idx
                        )
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        if (fx is not None) and (fy is not None) and (kappa is not None):
            viewpoint.update_calibration (fx, fy, kappa)
        
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        print(f"render frame: {cur_frame_idx}: fx {viewpoint.fx}, fy {viewpoint.fy}, kappa {viewpoint.kappa}, cam_center {viewpoint.camera_center}")
        return image, depth, opacity



    @staticmethod
    def __save_tensor2rgb (torch_tensor, filename="test.png"):
        image = torch_tensor.squeeze().squeeze()
        if not (image.ndim == 2 or image.ndim == 3):
            return
        if image.ndim == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        '''
            Torch Image: 3*M*N
            Numpy Image: M*N*3 [RGB]
            OpenCV stores images in BGR order instead of RGB
            plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        '''
        image = torch.clamp(image, min=0, max=1.0) * 255
        rgb = image.byte().permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(rgb).save(filename)


    def __save_depth_as_png(self, data, filename):
            """
            All depths are saved in np.uint16
            """
            data_np = data.data[0].cpu().float().numpy()
            data_np = data_np.astype(np.uint16)
            data_pil = Image.fromarray(np.squeeze(data_np), mode='I;16').convert(mode='I')

            data_pil.save(filename)


    def __generate_new_config_file(self, new_config_filename, output_data_dir):
        with open(self.config_file, "r") as yml:
            new_config = yaml.safe_load(yml)
        
        new_config["Dataset"]["dataset_path"] = output_data_dir
        new_config['Dataset']['SelfCalibration'] = {
            'enabled': True,
            'radial_distortion': 1,
            'frame_id': ", ".join(str(x) for x in self.idx_update),
            'gt_fx'   : ", ".join(str(x) for x in self.fx_update),
            'gt_kappa': ", ".join(str(x) for x in self.kappa_update),
            'backend_params': {
                    'lr_cnt1': 0.002,
                    'lr_cnt2': 0.0002                    
            }
        }
        new_config["Dataset"]["Calibration"]["distorted"] = False
        with open(new_config_filename, 'w') as file:
            yaml.safe_dump(new_config, file, sort_keys=False)




    def set_calibration(self, idx, fx, kappa):
        self.fx_arr[idx:] = fx
        self.fy_arr[idx:] = fx * self.aspect_ratio
        self.kappa_arr[idx:] = kappa

        self.idx_update.append(idx)
        self.fx_update.append(fx)
        self.fy_update.append(fx * self.aspect_ratio)
        self.kappa_update.append(kappa)



    def run(self, new_config_filename, output_data_dir):

        image_dir = os.path.join(output_data_dir, "rgb")
        depth_dir = os.path.join(output_data_dir, "depth")
        mkdir_p(image_dir)
        mkdir_p(depth_dir)

        for cur_frame_idx in range(0, len(self.dataset)):

            fx = self.fx_arr[cur_frame_idx]
            fy = self.fy_arr[cur_frame_idx]
            kappa = self.kappa_arr[cur_frame_idx]

            (image, depth, opacity ) = self.__render_impl(cur_frame_idx, fx, fy, kappa)

            color_path = self.color_paths[cur_frame_idx]
            head, tail = os.path.split(color_path)
            self.__save_tensor2rgb(image, filename=os.path.join(image_dir, tail) )

            """
            save depth image
            """
            if self.has_depth:
                depth_path = self.depth_paths[cur_frame_idx]
                head, tail = os.path.split(depth_path)
                filename=os.path.join(depth_dir, tail)

                depth *= self.depth_scale

                data_np = depth.data[0].cpu().float().numpy().astype(np.uint16)
                data_pil = Image.fromarray(np.squeeze(data_np), mode='I;16').convert(mode='I')
                data_pil.save(filename)

        """
        generate new config file
        """
        self.__generate_new_config_file(new_config_filename, output_data_dir)
        """
        copy txt files
        """
        dataset_path = self.config["Dataset"]["dataset_path"]
        for f in glob.glob(f"{dataset_path}/*.txt"):
            os.system(f"cp {f} {output_data_dir}") 
            print(f"copy file {f} to {output_data_dir}")




def run_tum_gt_pose():

    tum_configs = {}
    if True:
        tum_configs[1] = glob.glob(f"configs/mono/tum/fr1_desk.yaml")
        tum_configs[2] = glob.glob(f"configs/mono/tum/fr2_xyz.yaml")
        tum_configs[3] = glob.glob(f"configs/mono/tum/fr3_office.yaml")
    print(tum_configs)

    for _, configs in tum_configs.items():

        for config_file_path in configs:
            torch.cuda.empty_cache()
            command = f"python slam.py --config {config_file_path} --use_gt_pose --eval"
            print(f"Running: {command}")
            os.system(command)

    if os.path.exists("results/tum/fr1_desk_save"):
        os.system("rm -rf results/tum/fr1_desk_save")
    os.system("mv results/tum/fr1_desk results/tum/fr1_desk_save")

    if os.path.exists("results/tum/fr2_xyz_save"):
        os.system("rm -rf results/tum/fr2_xyz_save")
    os.system("mv results/tum/fr2_xyz results/tum/fr2_xyz_save")

    if os.path.exists("results/tum/fr3_office_save"):
        os.system("rm -rf results/tum/fr3_office_save")
    os.system("mv results/tum/fr3_office results/tum/fr3_office_save")





def main_tum_fr1():

    config_file = "configs/mono/tum/fr1_desk.yaml"
    result_path = "results/tum/fr1_desk_save"

    sim = SLAM_Dataset_Simulator(config_file=config_file, result_path=result_path)

    """
    set calibration changes
    """
    # sim.set_calibration(idx=100, fx=500, kappa=0.0)
    # sim.set_calibration(idx=200, fx=560, kappa=0.0)
    # sim.set_calibration(idx=300, fx=500, kappa=0.0)


    new_config_filename = "configs/mono/tum/fr1_desk_calib0.yaml"
    output_data_dir = "/hdd/slam/tum_calib/fr1_desk"    

    sim.run(new_config_filename, output_data_dir)



def main_tum_fr2():

    config_file = "configs/mono/tum/fr2_xyz.yaml"
    result_path = "results/tum/fr2_xyz_save"

    sim = SLAM_Dataset_Simulator(config_file=config_file, result_path=result_path)

    """
    set calibration changes
    """
    # sim.set_calibration(idx=100, fx=500, kappa=0.0)
    # sim.set_calibration(idx=200, fx=560, kappa=0.0)
    # sim.set_calibration(idx=300, fx=500, kappa=0.0)


    new_config_filename = "configs/mono/tum/fr2_xyz_calib0.yaml"
    output_data_dir = "/hdd/slam/tum_calib/fr2_xyz"    

    sim.run(new_config_filename, output_data_dir)



def main_tum_fr3():

    config_file = "configs/mono/tum/fr3_office.yaml"
    result_path = "results/tum/fr3_office_save"

    sim = SLAM_Dataset_Simulator(config_file=config_file, result_path=result_path)

    """
    set calibration changes
    """
    # sim.set_calibration(idx=100, fx=500, kappa=0.0)
    # sim.set_calibration(idx=200, fx=560, kappa=0.0)
    # sim.set_calibration(idx=300, fx=500, kappa=0.0)


    new_config_filename = "configs/mono/tum/fr3_office_calib0.yaml"
    output_data_dir = "/hdd/slam/tum_calib/fr3_office"    

    sim.run(new_config_filename, output_data_dir)





if __name__ == "__main__":
    run_tum_gt_pose()

    main_tum_fr1()
    main_tum_fr2()
    main_tum_fr3()

