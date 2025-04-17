from gaussian_splatting.gaussian_renderer import render
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from slam_impl.slam_utils import image_gradient, image_gradient_mask
from slam_impl.config_utils import load_config
from utils_cali.dataset_cali import load_dataset
from munch import munchify
from utils_cali.camera_cali_utils import CameraForCalibration as Camera
import cv2
# from pylatex import Document, Section, SubFigure, Figure, MiniPage, Command
import os
import json
import yaml
import csv
# from pylatex.utils import NoEscape
import pickle

from utils_cali.slam_cali_frontend import Simulator
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gui import gui_utils, slam_gui
import torch.multiprocessing as mp
from slam_impl.multiprocessing_utils import FakeQueue
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim

from evo.core import metrics, trajectory
from evo.tools import plot
from evo.core.trajectory import PosePath3D
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import copy
import open3d as o3d
# results_data = {
#     'monocular': {
#         'office2': {
#             'office2_vo':{
#                 '2024-10-10-11-54-59': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-54-59',
#                     'rmse': 1,
#                     'psnr':111,
#                     'single_thread': True,
#                 },
#                 '2024-10-10-11-25-50': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-25-50',
#                     'rmse': 0.5,
#                     'single_thread': False,
#                 },
#             },
#             'office2_v1':{

#             }
            
#         },
#         'office1': {
#             'office1_o':{
#                 '2024-10-10-11-54-59': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-54-59',
#                     'rmse': 0.6,
#                     'psnr':111,
#                     'single_thread': True,
#                 },
#                 '2024-10-10-11-25-50': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-25-50',
#                     'rmse': 0.7,
#                     'psnr':111,
#                     'single_thread': True,
#                 },
#             },
#             'office1_v1':{
#                 '2024-10-10-11-54-59': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-54-59',
#                     'rmse': 0.6,
#                     'psnr':111,
#                     'single_thread': True,
#                 },
#                 '2024-10-10-11-25-50': {
#                     'result_path': '/workspaces/src/MonoGS_dev/results/monocular/replica_small/office2/2024-10-10-11-25-50',
#                     'rmse': 0.7,
#                     'psnr':111,
#                     'single_thread': True,
#                 },
#             }
#         },
#     },
#     'depth': {

#     }
# }

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return None

def load_gs_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        return None

def load_pose(file_path):
    pass

def calculate_psnr(render_path, gt_path):
    def get_image(path):
        device = "cuda:0"
        dtype = torch.float32
        image = np.array(Image.open(path))
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=device, dtype=dtype)
        )
        return image

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to("cuda")
    gt_image = get_image(gt_path)
    render_image = get_image(render_path)

    image = torch.clamp(render_image, 0.0, 1.0)
    # save image
    gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    mask = gt_image > 0
    psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
    ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
    lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

    print("psnr: ", psnr_score.item())
    print("ssim: ", ssim_score.item())
    print("lpips: ", lpips_score.item())
    pass

class Experiment():
    def __init__(self, path):
        self.path = path
        self.final_stats_json_file_path = os.path.join(path, 'plot', 'stats_final.json')
        self.trj_final_json_file_path = os.path.join(path, 'plot', 'trj_final.json')
        self.before_opt_psnr_json_file_path = os.path.join(path, 'psnr', 'before_opt', 'final_result.json')
        self.after_opt_psnr_json_file_path = os.path.join(path, 'psnr', 'after_opt', 'final_result.json')
        self.yaml_file_path = os.path.join(path, 'config.yml')
        self.cali_result_json_file_path = os.path.join(path, 'cali', 'final_result.json')

        self.final_stats_json_data = read_json_file(self.final_stats_json_file_path)
        self.trj_final_json_data = read_json_file(self.trj_final_json_file_path)
        self.before_opt_psnr_json_data = read_json_file(self.before_opt_psnr_json_file_path)
        self.after_opt_psnr_json_data = read_json_file(self.after_opt_psnr_json_file_path)
        self.yaml_data = read_yaml_file(self.yaml_file_path)
        self.cali_result_json_data = read_json_file(self.cali_result_json_file_path)
    
    def iscompleted(self):
        return all([
            self.final_stats_json_data is not None,
            self.trj_final_json_data is not None,
            self.before_opt_psnr_json_data is not None,
            self.after_opt_psnr_json_data is not None,
            self.yaml_data is not None,
            self.cali_result_json_data is not None
        ])
    
    def load_data(self):
        rmse = self.final_stats_json_data.get('rmse', 'None')
        max_ate = self.final_stats_json_data.get('max', 'None')

        bef_opt_mean_psnr = self.before_opt_psnr_json_data.get('mean_psnr', 'None')
        bef_opt_mean_ssim = self.before_opt_psnr_json_data.get('mean_ssim', 'None')
        bef_opt_mean_lpips = self.before_opt_psnr_json_data.get('mean_lpips', 'None')

        aft_opt_mean_psnr = self.after_opt_psnr_json_data.get('mean_psnr', 'None')
        aft_opt_mean_ssim = self.after_opt_psnr_json_data.get('mean_ssim', 'None')
        aft_opt_mean_lpips = self.after_opt_psnr_json_data.get('mean_lpips', 'None')

        dataset_in_yaml_data = self.yaml_data['Dataset']
        sp = dataset_in_yaml_data.get('single_thread', 'None')
        # unit = self.dataset_in_yaml_data.get('unit', 'None')
        # pcd_downsample = self.dataset_in_yaml_data.get('pcd_downsample', 'None')
        # pcd_downsample_init = self.dataset_in_yaml_data.get('pcd_downsample_init', 'None')
        # point_size = self.dataset_in_yaml_data.get('point_size', 'None')
        calib_opts_allow_lens_distortion = self.yaml_data.get('calib_opts_allow_lens_distortion', 'None')
        calib_opts_require_calibration = self.yaml_data.get('calib_opts_require_calibration', 'None')

        after_mapping_itr_num = self.yaml_data.get('Training').get('after_mapping_itr_num', 'None')
        be_focal_lr = self.yaml_data.get('Training').get('be_focal_lr', 'None')
        be_focal_lr_cnt_s2 = self.yaml_data.get('Training').get('be_focal_lr_cnt_s2', 'None')
        afle = self.cali_result_json_data.get('AFLE', 'None')

        data = {
            'result_path': self.path,
            # 'unit': unit,
            # 'pcd_downsample': pcd_downsample,
            # 'pcd_downsample_init': pcd_downsample_init,
            # 'point_size': point_size,
            # 'edge_threshold':edge_threshold,
            # 'kf_translation': kf_translation,
            # 'kf_min_translation': kf_min_translation,
            'max': max_ate,
            'rmse': rmse,
            "afle": afle,
            'after_opt_mean_psnr': aft_opt_mean_psnr,
            'after_opt_mean_ssim': aft_opt_mean_ssim,
            'after_opt_mean_lpips': aft_opt_mean_lpips,
            'single_thread': sp,
            'after_mapping_itr_num': after_mapping_itr_num,
            'be_focal_lr_cnt_s2': be_focal_lr_cnt_s2,
            'be_focal_lr': be_focal_lr,
            'calib_opts_require_calibration': calib_opts_require_calibration,
            'calib_opts_allow_lens_distortion': calib_opts_allow_lens_distortion
        }
        return data

    def load_ply(self):
        self.ply_path = os.path.join(self.path, 'point_cloud', 'final', 'point_cloud.ply')
        self.ply_path = '/datasets/office1_mesh.ply'

        self.pcd = o3d.io.read_point_cloud(self.ply_path)
        pass
    def load_gt_ply(self):
        self.gt_ply_path = '/datasets/office1_mesh.ply'
        self.gt_pcd = o3d.io.read_point_cloud(self.ply_path)
        pass
    
    def load_intrinsics(self):
        # intrinsic_filename
        # dataset_path
        self.intrinsics_path = read_json_file(os.path.join(self.path, 'cali', 'final_result.json'))
        if self.intrinsics is None:
            return False
        return True

    def compare_ply(self):
        # Debug
        self.load_ply()
        self.load_gt_ply()
        # downsample
        self.pcd_down = self.pcd.voxel_down_sample(voxel_size=0.05)
        self.gt_pcd_down = self.gt_pcd.voxel_down_sample(voxel_size=0.05)
        # compareing the two point clouds fitness
        threshold = 30  # Max distance for correspondence points (adjust as needed)
        icp_result = o3d.pipelines.registration.registration_icp(
            self.pcd_down, self.gt_pcd_down, threshold,
            np.eye(4),  # Initial alignment estimate
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        print(icp_result)
        # Output the alignment results: fitness and inlier RMSE
        print(f"Fitness score: {icp_result.fitness}")  # Fraction of matching points
        print(f"Inlier RMSE: {icp_result.inlier_rmse}")

    def load_model(self):
        self.gaussians = load_gs_model(os.path.join(self.path, 'gs', 'instance.pkl'))
        if self.gaussians is None:
            return False
        return True

    def load_pose(self):
        poses = read_json_file(os.path.join(self.path, 'plot', 'trj_final.json'))
        if poses is None:
            return False
        self.trj_id = poses['trj_id']
        np_trj_est = np.array(poses['trj_est'])
        np_trj_gt = np.array(poses['trj_gt'])
        print(np_trj_est.shape)
        assert len(self.trj_id) == len(np_trj_est) == len(np_trj_gt)
        # list to tensor
        self.trj_est_torch = []
        self.trj_est_np = []
        self.trj_gt_torch = []
        self.trj_gt_np = []
        for i in range(len(self.trj_id)):
            T = np_trj_est[i, :, :]
            cam_pose_est = np.linalg.inv(T)
            self.trj_est_np.append(T)
            self.trj_est_torch.append(torch.tensor(cam_pose_est, device="cuda"))
            T = np_trj_gt[i, :, :]
            cam_pose_gt = np.linalg.inv(T)
            self.trj_gt_np.append(T)
            self.trj_gt_torch.append(torch.tensor(cam_pose_gt, device="cuda"))
        return True

    def load_config(self):
        self.config = load_config(self.yaml_file_path)
        model_params = munchify(self.config["model_params"])
        opt_params = munchify(self.config["opt_params"])
        pipeline_params = munchify(self.config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        self.dataset = load_dataset(model_params, model_params.source_path, config=self.config)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.mono =  True if self.yaml_data['Dataset']['sensor_type'] == 'monocular' else False

    def load_focal(self):
        focal = read_json_file(os.path.join(self.path, 'cali', 'final_result.json'))
        if focal is None:
            return False
        self.focal_est = focal['focal_est']
        self.focal_gt = focal['focal_gt']
        self.kappa_est = focal['kappa_est']
        self.kappa_gt = focal['kappa_gt']
        self.focal_error_percentage = focal['focal_percentage']
        self.focal_error_percentage = 100 * np.array(self.focal_error_percentage)
        return True
                              
    def render(self):
        self.load_config()
        self.load_pose()
        self.load_model()
        self.load_focal()
        if self.load_pose() and self.load_model() and self.load_focal():
            if len(self.trj_id) == len(self.focal_est) == len(self.focal_gt):
                dir = os.path.join(self.path, 'rendering')
                os.makedirs(os.path.join(dir, 'pred'), exist_ok=True)
                os.makedirs(os.path.join(dir, 'gt'), exist_ok=True)
                img_pred, img_gt, saved_frame_idx = [], [], []
                psnr_array, ssim_array, lpips_array = [], [], []
                cal_lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).to("cuda")
                for i in range(len(self.trj_id)):
                    idx = self.trj_id[i]
                    gt_image, _, _ = self.dataset[idx]
                    viewpoint = Camera.init_from_dataset(self.dataset, idx)
                    viewpoint.compute_grad_mask(self.config)

                    viewpoint.R = self.trj_est_torch[i][:3, :3]
                    viewpoint.T = self.trj_est_torch[i][:3, 3]
                    viewpoint.update_calibration(self.focal_est[i], self.focal_est[i], self.kappa_est[i])
                    # viewpoint.fx = self.focal_gt[i]
                    # viewpoint.fy = self.focal_gt[i]
                    # viewpoint.kappa = self.kappa_gt[i]
                    # viewpoint.R = self.trj_gt[i][:3, :3]
                    # viewpoint.T = self.trj_gt[i][:3, 3]

                    # rendering_result = eval_rendering(
                    #     self.frontend.cameras,
                    #     self.gaussians,
                    #     self.dataset,
                    #     self.save_dir,
                    #     self.pipeline_params,
                    #     self.background,
                    #     kf_indices=kf_indices,
                    #     iteration="after_opt",
                    # )
                    rendering = render(viewpoint, self.gaussians, self.pipeline_params, self.background)["render"]
                    image = torch.clamp(rendering, 0.0, 1.0)
                    # save image
                    gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

                    # print(gt.shape)  # Should show a tuple of (height, width, channels) for color or (height, width) for grayscale
                    # print(gt.dtype)  # Should typically be 'uint8' for image data
                    # print(os.path.join(dir, f'gt/image_gt_{idx}.png'))
                    cv2.imwrite(os.path.join(dir, f'pred/image_pred_{idx}.png'), pred)
                    state = cv2.imwrite(os.path.join(dir, f'gt/image_gt_{idx}.png'), gt)


                    mask = gt_image > 0
                    psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
                    ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
                    lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

                    psnr_array.append(psnr_score.item())
                    ssim_array.append(ssim_score.item())
                    lpips_array.append(lpips_score.item())
                    # path/pred/image_pred_{i}.png

                print('mean psnr:', np.mean(psnr_array))
                print('mean ssim:', np.mean(ssim_array))
                print('mean lpips:', np.mean(lpips_array))
                # write mean psnr, ssim, lpips to a file
                with open(os.path.join(dir, 'psnr_final_result.json'), 'w') as file:
                    json.dump({
                        'mean_psnr': np.mean(psnr_array),
                        'mean_ssim': np.mean(ssim_array),
                        'mean_lpips': np.mean(lpips_array),
                        'psnr_array': psnr_array,
                        'ssim_array': ssim_array,
                        'lpips_array': lpips_array
                    }, file)
                return rendering

    def pre_plot(self):
        self.load_config()
        self.load_pose()
        self.load_model()
        self.load_focal()

        if self.load_pose() and self.load_model() and self.load_focal():
            dir = os.path.join(self.path, 'cali')
            self.traj_ref = PosePath3D(poses_se3=self.trj_gt_np)
            self.traj_est = PosePath3D(poses_se3=self.trj_est_np)
            self.traj_est_aligned = copy.deepcopy(self.traj_est)
            self.traj_est_aligned.align(self.traj_ref, self.mono)
            return True

    def plot(self):
        if self.pre_plot():
            # plot the trajectory with focal length error bar
            dir = os.path.join(self.path, 'cali')
            plot_mode = PlotMode.xy
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            ax.set_title(f"Focal length error (%)")
            plot.traj(ax, plot_mode, self.traj_ref, "--", "grey", "gt")
            plot.traj_colormap(
                ax,
                self.traj_est_aligned,
                self.focal_error_percentage,
                plot_mode,
                min_map=min(self.focal_error_percentage),
                max_map=max(self.focal_error_percentage),
            )
            # plt.show()
            plt.savefig(os.path.join(dir, "traj_focal_error.png"), dpi=90)

        pass
     

class Results():
    def __init__(self, datasets, img_types=['monocular','depth'], seqs=['o0','o1', 'o2','o3','o4']):
        self.datasets = datasets # must be a list, asserts  ['replica_small', 'replica_small_cali']
        # data structrur
        # self.data[img_type][seq][sub_seq][time]={'path':'xxx','rmse':'xxx'}
        self.data = {} 
        self.generate_base_path(img_types)
        self.generate_seq_path(seqs)
        self.generate_data()
        # self.tracking_latex_table()

    def generate_base_path(self, img_types):
        self.img_types = []
        self.base_paths = []
        for img_type in img_types:
            if img_type == 'mono' or img_type == 'rgb' or img_type == 'monocular':
                self.img_types.append('monocular')
            elif img_type == 'rgbd' or img_type == 'depth':
                self.img_types.append('depth')
        for j in self.img_types:
            self.data[j]={}
            for i in self.datasets:
                self.base_paths.append('/workspaces/src/MonoGS_dev/results/' + j + '/'+ i)
    
    def generate_seq_path(self, seqs):
        self.seqs_paths = []
        # Create a mapping to convert 'oX' to 'officeX'
        # Create a mapping to convert 'oX_600' to 'officeX_600'
        seq_mapping = []
        for s in seqs:
            if s.startswith('o'):  # Short form like 'o0'
                seq_mapping.append(f'office{s[-1]}')
                for img_type in self.img_types:
                    self.data[img_type][f'office{s[-1]}'] = {} 
        for base_path in self.base_paths:
            for sequence_dir in os.listdir(base_path):
                if any(mapping in sequence_dir for mapping in seq_mapping):
                    sequence_path = os.path.join(base_path, sequence_dir)
                    self.seqs_paths.append(sequence_path)

    def generate_data(self):
        for sequence_dir in self.seqs_paths:
            parts = sequence_dir.split('/')
            # self.data[img_type][seq][sub_seq][time]={'path':'xxx','rmse':'xxx'}
            img_type = parts[-3]
            seq = parts[-1][:7]
            sub_seq = parts[-1]
            self.data[img_type][seq][sub_seq] = {}
            for time in os.listdir(sequence_dir):
                data_path = os.path.join(sequence_dir, time)
                # print(data_path)
                experiment = Experiment(data_path)
                experiment.load_model()
                gaussians = experiment.gaussians

                if experiment.iscompleted():
                    data = experiment.load_data()
                    self.data[img_type][seq][sub_seq][time] = data
                    if gaussians is not None:
                        # experiment.render()
                        # experiment.plot()
                        pass
    
    def tracking_latex_table(self):
        tables = {}
        # self.data[img_type][seq][sub_seq][time]={'path':'xxx','rmse':'xxx'}
        for img_type, sequences in self.data.items():
            for seq, sub_seqs in sequences.items():
                sorted_sub_seqs = sorted(sub_seqs.keys())
                lr = set()
                # print(seq, sub_seqs)
                # Start building the LaTeX table for this sequence
                latex_code = "\\begin{table}[ht]\n\\centering\n"
                # Dynamically create columns based on the sub-sequences and add two extra columns for RMSE and Path
                latex_code += "\\begin{tabular}{|" + "c|"*(len(sub_seqs)+2) + "}\n"
                # latex_code += "Types & Methods & o0 & o0-v0 & o0-v1 & o0-v2 & o0-v3 & o0-v4 \\\\\n"
                latex_code += "\\hline\n"

                # Header line with sub-sequence names repeated for each metric
                header_line = "Types & Methods"

                for sub_seq in sorted_sub_seqs:
                    header_line += " & " + f"{sub_seq.replace('_', '-')}"
                # header_line = header_line.rstrip("& ")  # Remove the last extra ampersand                
                latex_code += header_line + " \\\\\\hline\n"
                # print(latex_code)
                adjusted_data_dict = {}
                for sub_seq in sorted_sub_seqs:
                    for time, data in sub_seqs[sub_seq].items():
                        key = 'True' if data['single_thread'] else 'False'
                        if key not in adjusted_data_dict:
                            adjusted_data_dict[key] = []
                        # if len(lr) ==0:
                        #     lr.add(data['be_focal_lr'])
                        #     adjusted_data_dict[key].append((sub_seq, 100* data['rmse']))
                        # elif  len(lr) !=0 and data['be_focal_lr'] in lr:
                        #     adjusted_data_dict[key].append((sub_seq, 100* data['rmse']))
                        adjusted_data_dict[key].append((sub_seq, 100* data['rmse']))
                        lr.add(data['be_focal_lr'])
                # print(adjusted_data_dict)

                for key, values in adjusted_data_dict.items():
                    rmse_line = ''
                    type = "RGB" if img_type == "monocular" else "RGBD"
                    if key == 'True':
                        method = 'CaliGS-SLAM(sp)'
                        # print(values.sequence_id)
                    else:
                        method = 'CaliGS-SLAM'
                    values = sorted(values, key=lambda x: x[0])
                    for seq_id, rmse in values:
                        rmse_line += f" & {rmse:.2f}"
                    latex_code += type + " & " + method + rmse_line + "\\\\\n"
                
                # Collect all unique timestamps across sub-sequences

                # Close the LaTeX table
                latex_code += "\\end{tabular}\n"
                latex_code += f"\\caption{{Camera tracking result on Replica. ATE RMSE in cm. Results for {seq} in {img_type} with lr {lr}}}\n"
                latex_code += f"\\label{{tab:{img_type}_{seq}}}\n"
                latex_code += "\\end{table}\n"
                
                # Store the table in a dictionary with keys as sequence names
                tables[f"{img_type}_{seq}"] = latex_code

                print(latex_code)

        return tables
    
    def rendering_latex_table(self):
        tables = {}

        # self.data[img_type][seq][sub_seq][time]={'path':'xxx','rmse':'xxx'}
        for img_type, sequences in self.data.items():
            for seq, sub_seqs in sequences.items():
                sorted_sub_seqs = sorted(sub_seqs.keys())
                be_focal_lr_values = set()
                be_focal_lr_cnt_s2_values = set()
                adjusted_data_dict = {}
                lr = set()

                latex_code = "\\begin{table}[ht]\n"
                latex_code += "\\centering\n"
                latex_code += "\\begin{tabular}{|" + "c|"*(6) + "}\n"

                latex_code += "\\hline\n"
                # latex_code += seq_line + "\\\\\n"
                latex_code += "Types & Seq & Methods & PSNR[db]$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\\n"

                for sub_seq in sorted_sub_seqs:
                    for time, data in sub_seqs[sub_seq].items():
                        key = 'True' if data['single_thread'] else 'False'
                        type = "RGB" if img_type == "monocular" else "RGBD"
                        if key == 'True':
                            method = 'CaliGS-SLAM(sp)'
                            # print(values.sequence_id)
                        else:
                            method = 'CaliGS-SLAM'
                        lr.add(data['be_focal_lr'])

                        rendering_line = f" & {data['after_opt_mean_psnr']:.2f} & {data['after_opt_mean_ssim']:.2f} & {data['after_opt_mean_lpips']:.3f}"
                        latex_code += type + " & " + method + " & " + sub_seq.replace('_', '-') + f" {data['be_focal_lr']}" + f" {data['calib_opts_require_calibration']}" + rendering_line + "\\\\\n"


                latex_code += "\\hline\n"
                latex_code += "\\end{tabular}\n"
                latex_code += f"\\caption{{Camera rendering result on Replica. Results for {seq} in {img_type} with lr {lr}}}\n"
                latex_code += "\\label{tab:rendering}\n"
                latex_code += "\\end{table}\n"  

                tables[f"{img_type}_{seq}"] = latex_code

                print(latex_code)

        return tables
    
    def total_latex_table(self):
        tables = {}

        # self.data[img_type][seq][sub_seq][time]={'path':'xxx','rmse':'xxx'}
        for img_type, sequences in self.data.items():
            for seq, sub_seqs in sequences.items():
                sorted_sub_seqs = sorted(sub_seqs.keys())
                be_focal_lr_values = set()
                be_focal_lr_cnt_s2_values = set()
                adjusted_data_dict = {}
                lr = set()

                latex_code = "\\begin{table}[ht]\n"
                latex_code += "\\centering\n"
                latex_code += "\\begin{tabular}{|" + "c|"*(8) + "}\n"

                latex_code += "\\hline\n"
                # latex_code += seq_line + "\\\\\n"
                latex_code += "Types & Seq & Methods & RMSE[cm] & AFEL & PSNR[db]$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\\n"

                for sub_seq in sorted_sub_seqs:
                    for time, data in sub_seqs[sub_seq].items():
                        key = 'True' if data['single_thread'] else 'False'
                        type = "RGB" if img_type == "monocular" else "RGBD"
                        if key == 'True':
                            method = 'CaliGS-SLAM(sp)'
                            # print(values.sequence_id)
                        else:
                            method = 'CaliGS-SLAM'
                        lr.add(data['be_focal_lr'])
                        total_line =  f" & {100*data['rmse']:.3f}" + f" & {data['afle']:.3f}" + f" & {data['after_opt_mean_psnr']:.4f} & {data['after_opt_mean_ssim']:.4f} & {data['after_opt_mean_lpips']:.4f}"
                        # latex_code += type + " & " + method + " & " + sub_seq.replace('_', '-') + f" {data['be_focal_lr']}" + rendering_line + "\\\\\n"
                        latex_code += type + " & " + method + " & " + sub_seq.replace('_', '-') + f" {data['be_focal_lr']}" + f" {data['calib_opts_require_calibration']}" + total_line + "\\\\\n"


                latex_code += "\\hline\n"
                latex_code += "\\end{tabular}\n"
                latex_code += f"\\caption{{Camera tracking result on Replica. ATE RMSE in cm. Results for {seq} in {img_type} with lr {lr}}}\n"
                latex_code += "\\label{tab:rendering}\n"
                latex_code += "\\end{table}\n"  

                tables[f"{img_type}_{seq}"] = latex_code

                print(latex_code)

        return tables
    
    def generate_table1(self):
        # ensure there is o0-o4 and ours
        # ensure there is monocular and depth
        # ensure there is sp and non-sp
        # for 
        pass


    def plot_gs(self):
        pass

def plot_traj_focal(ours, ours_no_cali, gt):
    plot_mode = PlotMode.xy
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"Focal length error (%)")
    plot.traj(ax, plot_mode, gt.traj_ref, "--", "grey", "gt")
    # plot.traj(ax, plot_mode, gsslam.traj_est_aligned, "--", "r", "gt")
    min_map = min(ours.focal_error_percentage, ours_no_cali.focal_error_percentage)
    max_map = max(ours.focal_error_percentage, ours_no_cali.focal_error_percentage)
    plot.traj_colormap(
        ax,
        ours.traj_est_aligned,
        ours.focal_error_percentage,
        plot_mode,
        min_map=min_map,
        max_map=max_map,
    )
    plot.traj_colormap(
        ax,
        ours_no_cali.traj_est_aligned,
        ours_no_cali.focal_error_percentage,
        plot_mode,
        min_map=min_map,
        max_map=min_map,
    )
    # plt.show()
    plt.savefig(os.path.join("traj_focal_error.png"), dpi=90)

if __name__ == "__main__":
    img_types = ['mono', 'rgbd']
    datasets = ['replica_small', 'replica_small_cali']
    # sequence = ['o0','o1', 'o2','o3','o4']


    # img_types = ['mono']
    # datasets = ['replica_small']
    # sequence = ['o0','o1', 'o2','o3','o4']
    sequence = ['o4']
    results = Results(datasets, img_types, sequence)
    # tables = results.tracking_latex_table()
    # tables = results.rendering_latex_table()
    tables = results.total_latex_table()
    # a = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small/office0/2024-10-24-10-04-59')
    # a = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small/office3/2024-11-09-22-26-48')
    # # a.render()
    # a.plot()

    # gsslam = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small/office0/2024-11-08-18-38-01')
    # ours_no_cali = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small_cali/office0_v6/2024-11-07-22-15-28')
    # ours.pre_plot()
    # gsslam.pre_plot()
    # ours = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small_cali/office0_v6/2024-11-09-11-57-25')
    # ours = Experiment('/workspaces/src/MonoGS_dev/results/monocular/replica_small/office1_1000/2024-11-09-21-01-20')
    # ours.compare_ply()
    # ours.pre_plot()
    # plot_traj_focal(ours, ours_no_cali, gsslam)

    # dir = "/datasets/after_opt+gt/"
    # # read txt
    # with open(dir + "frame_id.txt", "r") as file:
    #     # remove \n at the end of each line
    #     gt = [line.strip() for line in file]
    # for indx in gt:
    #     render_img_path = dir + "after_opt/" + f"render_{indx}.png"
    #     gt_img_path = dir + "gt/" + f"frame{int(indx):06d}.jpg"
    #     # if file exists
    #     if os.path.exists(render_img_path):
    #         print(indx)
    #         calculate_psnr(render_img_path, gt_img_path)
        

    
    pass