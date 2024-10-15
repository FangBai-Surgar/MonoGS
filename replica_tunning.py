import os
import json
import yaml
import csv
import numpy as np

# plot
import matplotlib.pyplot as plt
from evo.core import metrics, trajectory
from evo.tools import plot
from evo.core.trajectory import PosePath3D
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

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

def collect_data_from_directory(base_path):
    data_dict = {}
    trj_dict = {}

    for sequence_dir in os.listdir(base_path):
        sequence_path = os.path.join(base_path, sequence_dir)
        sequence = sequence_dir.split('/')[-1]
        print(sequence)
        # if sequence != 'office0':
            # continue
        # if sequence inlcude 4:
        # if 'office4' in sequence:
        #     continue
        if os.path.isdir(sequence_path):
            data_dict[sequence] = {}
            trj_dict[sequence] = {}
            for date_dir in os.listdir(sequence_path):
                date_path = os.path.join(sequence_path, date_dir)
                if os.path.isdir(date_path):
                    final_stats_json_file_path = os.path.join(date_path, 'plot', 'stats_final.json')
                    trj_final_json_file_path = os.path.join(date_path, 'plot', 'trj_final.json')
                    before_opt_psnr_json_file_path = os.path.join(date_path, 'psnr', 'before_opt', 'final_result.json')
                    after_opt_psnr_json_file_path = os.path.join(date_path, 'psnr', 'after_opt', 'final_result.json')
                    yaml_file_path = os.path.join(date_path, 'config.yml')

                    final_stats_json_data = read_json_file(final_stats_json_file_path)
                    trj_final_json_data = read_json_file(trj_final_json_file_path)
                    before_opt_psnr_json_data = read_json_file(before_opt_psnr_json_file_path)
                    after_opt_psnr_json_data = read_json_file(after_opt_psnr_json_file_path)
                    yaml_data = read_yaml_file(yaml_file_path)

                    if final_stats_json_data and yaml_data and before_opt_psnr_json_data and after_opt_psnr_json_data:
                        rmse = final_stats_json_data.get('rmse', 'None')

                        bef_opt_mean_psnr = before_opt_psnr_json_data.get('mean_psnr', 'None')
                        bef_opt_mean_ssim = before_opt_psnr_json_data.get('mean_ssim', 'None')
                        bef_opt_mean_lpips = before_opt_psnr_json_data.get('mean_lpips', 'None')

                        aft_opt_mean_psnr = after_opt_psnr_json_data.get('mean_psnr', 'None')
                        aft_opt_mean_ssim = after_opt_psnr_json_data.get('mean_ssim', 'None')
                        aft_opt_mean_lpips = after_opt_psnr_json_data.get('mean_lpips', 'None')

                        dataset_in_yaml_data = yaml_data['Dataset']
                        sp = dataset_in_yaml_data.get('single_thread', 'None')
                        # unit = dataset_in_yaml_data.get('unit', 'None')
                        # pcd_downsample = dataset_in_yaml_data.get('pcd_downsample', 'None')
                        # pcd_downsample_init = dataset_in_yaml_data.get('pcd_downsample_init', 'None')
                        # point_size = dataset_in_yaml_data.get('point_size', 'None')
                        calib_opts_allow_lens_distortion = yaml_data.get('calib_opts_allow_lens_distortion', 'None')
                        calib_opts_require_calibration = yaml_data.get('calib_opts_require_calibration', 'None')

                        after_mapping_itr_num = yaml_data.get('Training').get('after_mapping_itr_num', 'None')
                        be_focal_lr = yaml_data.get('Training').get('be_focal_lr', 'None')
                        be_focal_lr_cnt_s2 = yaml_data.get('Training').get('be_focal_lr_cnt_s2', 'None')


                        # edge_threshold = yaml_data.get('Training').get('edge_threshold', 'None')
                        # kf_translation = yaml_data.get('Training').get('kf_translation', 'None')
                        # kf_min_translation = yaml_data.get('Training').get('kf_min_translation', 'None')


                        data_dict[sequence][date_dir] = {
                            'result_path': date_path,
                            # 'unit': unit,
                            # 'pcd_downsample': pcd_downsample,
                            # 'pcd_downsample_init': pcd_downsample_init,
                            # 'point_size': point_size,
                            # 'edge_threshold':edge_threshold,
                            # 'kf_translation': kf_translation,
                            # 'kf_min_translation': kf_min_translation,
                            'rmse': rmse,
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

                    if trj_final_json_data:
                        trj_data = dict()
                        trj_data['trj_id'] = trj_final_json_data.get('trj_id', 'None')
                        trj_data['trj_est'] = trj_final_json_data.get('trj_est', 'None')
                        trj_data['trj_gt']  = trj_final_json_data.get('trj_gt', 'None')
                        trj_dict[sequence][date_dir] = trj_data
                        
    return data_dict, trj_dict

def write_to_csv(data_records, output_path):
    # Define the fieldnames for the CSV file
    fieldnames = [
        'sequence_id', 
        # 'unit', 'pcd_downsample', 'pcd_downsample_init', 'point_size', 'edge_threshold', 'kf_translation', 'kf_min_translation',
        'rmse[m]', 'after_opt_mean_psnr', 'after_opt_mean_ssim', 'after_opt_mean_lpips', 
        'sp',
        'after_mapping_itr_num', 'be_focal_lr_cnt_s2', 'be_focal_lr',
        # 'timestamp', 
        'calib_opts_require_calibration', 'calib_opts_allow_lens_distortion',
        'result_path'
    ]
    
    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Iterate over each sequence and its data records
        for sequence_id, timestamps in data_records.items():
            for timestamp, data in timestamps.items():
                # Create a row with required data
                row = {
                    'sequence_id': sequence_id,
                    # 'unit': data['unit'],
                    # 'pcd_downsample': data['pcd_downsample'],
                    # 'pcd_downsample_init': data['pcd_downsample_init'],
                    # 'point_size': data['point_size'],
                    # 'edge_threshold': data['edge_threshold'],
                    # 'kf_translation': data['kf_translation'],
                    # 'kf_min_translation': data['kf_min_translation'],
                    'rmse[m]': data['rmse'],
                    'after_opt_mean_psnr': data['after_opt_mean_psnr'],
                    'after_opt_mean_ssim': data['after_opt_mean_ssim'],
                    'after_opt_mean_lpips': data['after_opt_mean_lpips'],
                    'sp': data['single_thread'],
                    'after_mapping_itr_num': data['after_mapping_itr_num'],
                    'be_focal_lr_cnt_s2': data['be_focal_lr_cnt_s2'],
                    'be_focal_lr': data['be_focal_lr'],
                    # 'timestamp': timestamp,
                    'calib_opts_require_calibration': data['calib_opts_require_calibration'],
                    'calib_opts_allow_lens_distortion': data['calib_opts_allow_lens_distortion'],
                    'result_path': data['result_path'],
                }
                # Write the data row to the CSV file
                writer.writerow(row)

def plot_correlation(traj_ref, traj_est_aligned, plotMode, ax, interval):
    if plotMode == PlotMode.xy:
        for i, (ref_pose, est_pose) in enumerate(zip(traj_ref.poses_se3, traj_est_aligned.poses_se3)):
            if i % interval == 0:
                x_values = [ref_pose[0, 3], est_pose[0, 3]]
                y_values = [ref_pose[1, 3], est_pose[1, 3]]
                ax.plot(x_values, y_values, color="gray", linestyle="--", alpha=0.5)
    elif plotMode == PlotMode.xyz:
        for i, (ref_pose, est_pose) in enumerate(zip(traj_ref.poses_se3, traj_est_aligned.poses_se3)):
            if i % interval == 0:
                x_values = [ref_pose[0, 3], est_pose[0, 3]]  # X coordinates of the poses
                y_values = [ref_pose[1, 3], est_pose[1, 3]]  # Y coordinates of the poses
                z_values = [ref_pose[2, 3], est_pose[2, 3]]
                ax.plot(x_values, y_values, z_values, color="gray", linestyle="--", alpha=0.5)
    elif plotMode == PlotMode.yz:
        for i, (ref_pose, est_pose) in enumerate(zip(traj_ref.poses_se3, traj_est_aligned.poses_se3)):
            if i % interval == 0:
                y_values = [ref_pose[1, 3], est_pose[1, 3]]
                z_values = [ref_pose[2, 3], est_pose[2, 3]]
                ax.plot(y_values, z_values, color="gray", linestyle="--", alpha=0.5)
    return ax

def plot_pose_axes(ax, pose, scale=0.1):
    # Orientation is in the upper left 3x3 part of the pose matrix
    orientation = pose[:3, :3]
    position = pose[:3, 3]

    # Each column of the rotation matrix represents the direction of an axis
    x_dir, y_dir, z_dir = orientation[:, 0], orientation[:, 1], orientation[:, 2]

    # Plotting arrows for the x, y, and z directions
    ax.quiver(position[0], position[1], position[2], x_dir[0], x_dir[1], x_dir[2], 
              color='red', length=scale, normalize=True)
    ax.quiver(position[0], position[1], position[2], y_dir[0], y_dir[1], y_dir[2], 
              color='green', length=scale, normalize=True)
    ax.quiver(position[0], position[1], position[2], z_dir[0], z_dir[1], z_dir[2], 
              color='blue', length=scale, normalize=True)
    return ax

def plot_evo(trj_records):
    for sequence_id, timestamps in trj_records.items():
        for timestamp, trj_data in timestamps.items():
            trj_data_dict = trj_records[sequence_id][timestamp]

            trj_est_list = trj_data_dict['trj_est']
            trj_gt_list = trj_data_dict['trj_gt']
            trj_id_list = trj_data_dict['trj_id']

            trj_gt_np = [np.array(pose) for pose in trj_gt_list]
            trj_est_np = [np.array(pose) for pose in trj_est_list]


            traj_ref = PosePath3D(poses_se3=trj_gt_np)
            traj_est = PosePath3D(poses_se3=trj_est_np)
            traj_est_aligned, r_a, t_a, s = trajectory.align_trajectory(    
                traj=traj_est, 
                traj_ref=traj_ref,
                correct_scale=True, 
                return_parameters=True
            )
            print(f"Alignment parameters: r_a={r_a}, t_a={t_a}, s={s}")
            pose_relation = metrics.PoseRelation.translation_part
            data = (traj_ref, traj_est_aligned)
            ape_metric = metrics.APE(pose_relation)
            rpe_metric = metrics.RPE(pose_relation)
            ape_metric.process_data(data)
            ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
            ape_stats = ape_metric.get_all_statistics()
            plot_mode = PlotMode.xyz
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            ax.set_title(f"{sequence_id} timestamp: {timestamp}")
            plot.traj(ax, plot_mode, traj_ref, "--", "r", "gt")
            plot.traj_colormap(
                ax,
                traj_est_aligned,
                ape_metric.error,
                plot_mode,
                min_map=ape_stats["min"],
                max_map=ape_stats["max"],
            )
            
            interval = 10
            ax = plot_correlation(traj_ref, traj_est_aligned, plot_mode, ax, interval)
            for i, pose in enumerate(traj_est_aligned.poses_se3):
                if i % interval == 0:
                    ax = plot_pose_axes(ax, pose, scale=0.01)

            # Plotting axes for every 10th pose in the reference trajectory
            for i, pose in enumerate(traj_ref.poses_se3):
                if i % interval == 0:
                    ax = plot_pose_axes(ax, pose, scale=0.01)
            plt.show()

def dict_to_rmse_latex(data_dict):
    # Check with each sequence_id, it has two timestamps
    be_focal_lr_values = set()
    be_focal_lr_cnt_s2_values = set()
    for sequence_id, timestamps in data_dict.items():
        assert len(timestamps) == 2, f"Sequence {sequence_id} does not have exactly two timestamps."
        single_thread_values = [data['single_thread'] for timestamp, data in timestamps.items()]
        # Check for one True and one False
        assert single_thread_values.count(True) == 1 and single_thread_values.count(False) == 1, \
        f"Sequence {sequence_id} does not have one True and one False for 'single_thread'."
        for timestamp, data in timestamps.items():
            be_focal_lr_values.add(data['be_focal_lr'])
            be_focal_lr_cnt_s2_values.add(data['be_focal_lr_cnt_s2'])
    # Check if there is only one unique value for each parameter
    assert len(be_focal_lr_values) == 1, "Different 'be_focal_lr' values found across the sequences."
    assert len(be_focal_lr_cnt_s2_values) == 1, "Different 'be_focal_lr_cnt_s2' values found across the sequences." 
    adjusted_data_dict = {}
    for sequence_id, timestamps in data_dict.items():
        for timestamp, data in timestamps.items():
            key = 'True' if data['single_thread'] else 'False'
            if key not in adjusted_data_dict:
                adjusted_data_dict[key] = []
            adjusted_data_dict[key].append((sequence_id, 100* data['rmse']))
    rmse_lines = []
    for key, values in adjusted_data_dict.items():
        if key == 'True':
            type = 'CaliGS-SLAM(sp)'
            # print(values.sequence_id)
        else:
            type = 'CaliGS-SLAM'
        # sorted values by its last element in sequence_id
        values = sorted(values, key=lambda x: x[0])
        rmse_line = " & " + type
        seq_line = "Types & Methods" 
        for seq_id, rmse in values:
            rmse_line += f" & {rmse:.4f}"
            # seq_line += f" {seq_id} & " 
            seq_line += f" & {seq_id.replace('_', '-')}"
        rmse_lines.append(rmse_line)
        print(seq_line)
        print(rmse_line)

    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|" + "c|"*(len(data_dict)+2) + "}\n"
    latex_code += "\\hline\n"
    latex_code += seq_line + "\\\\\n"
    # latex_code += "Types & Methods & o0 & o0-v0 & o0-v1 & o0-v2 & o0-v3 & o0-v4 \\\\\n"
    latex_code += "\hline \\hline\n"
    if 'sp' in rmse_lines[0]:
        # If 'sp' is in the first element, append it first
        latex_code += rmse_lines[1] + "\\\\\n"
        latex_code += "\\hline\n"
        latex_code += "\\multirow{-2}{*}{RGB}" + rmse_lines[0] + "\\\\\n"
    else:
        # If 'sp' is not in the first element, append the second element first
        latex_code += rmse_lines[0] + "\\\\\n"
        latex_code += "\\hline\n"
        latex_code += "\\multirow{-2}{*}{RGB}" + rmse_lines[1] + "\\\\\n"
    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{ATE RMSE in cm. different methods and sequences.}\n"
    latex_code += "\\label{tab:rmse}\n"
    latex_code += "\\end{table}\n"   
    return latex_code

def dict_to_psnr_latex(data_dict):
    # Check with each sequence_id, it has two timestamps
    be_focal_lr_values = set()
    be_focal_lr_cnt_s2_values = set()
    for sequence_id, timestamps in data_dict.items():
        assert len(timestamps) == 2, f"Sequence {sequence_id} does not have exactly two timestamps."
        single_thread_values = [data['single_thread'] for timestamp, data in timestamps.items()]
        # Check for one True and one False
        assert single_thread_values.count(True) == 1 and single_thread_values.count(False) == 1, \
        f"Sequence {sequence_id} does not have one True and one False for 'single_thread'."
        for timestamp, data in timestamps.items():
            be_focal_lr_values.add(data['be_focal_lr'])
            be_focal_lr_cnt_s2_values.add(data['be_focal_lr_cnt_s2'])
    # Check if there is only one unique value for each parameter
    assert len(be_focal_lr_values) == 1, "Different 'be_focal_lr' values found across the sequences."
    assert len(be_focal_lr_cnt_s2_values) == 1, "Different 'be_focal_lr_cnt_s2' values found across the sequences." 
    adjusted_data_dict = {}
    for sequence_id, timestamps in data_dict.items():
        print(sequence_id)
        for timestamp, data in timestamps.items():
            key = 'True' if data['single_thread'] else 'False'
            if sequence_id not in adjusted_data_dict:
                adjusted_data_dict[sequence_id] = []
            adjusted_data_dict[sequence_id].append((key, data['after_opt_mean_psnr'], data['after_opt_mean_ssim'], data['after_opt_mean_lpips']))

    lines = []
    # sort the dictionary by sequence_id
    seqs = sorted(adjusted_data_dict.keys())
    
    for sequence_id in seqs:
        values = adjusted_data_dict[sequence_id]
        # print(values)
        for sp, psnr, ssim, lpips in values:
            print(sp, psnr, ssim, lpips)
            if sp == 'True':
                key = 'CaliGS-SLAM(sp)'
            else:
                key = 'CaliGS-SLAM'
            line = f" {sequence_id} & {key} & {psnr:.4f} & {ssim:.4f} & {lpips:.4f}".replace('_', '-')
            # print(line)
            lines.append(line)
    
    latex_code = "\\begin{table}[ht]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|" + "c|"*(5) + "}\n"

    latex_code += "\\hline\n"
    # latex_code += seq_line + "\\\\\n"
    latex_code += "Seq & Methods & PSNR[db]$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ \\\\\n"
    latex_code += "\hline \\hline\n"
    for i in lines:
        # If 'sp' is in the first element, append it first
        latex_code += i + "\\\\\n"
        latex_code += "\\hline\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{ATE RMSE in cm. different methods and sequences.}\n"
    latex_code += "\\label{tab:rmse}\n"
    latex_code += "\\end{table}\n"   
    return latex_code

    # for sequence_id, timestamps in data_dict.items():
    #     first_timestamp = True
    #     for date_dir, metrics in timestamps.items():
    #         if first_timestamp:
    #             latex_code += f"        \\multirow{{2}}{{*}}{{{sequence_id}}} & {date_dir} & {metrics['rmse']} & {metrics['after_opt_mean_psnr']} & {metrics['after_opt_mean_ssim']} & {'Yes' if metrics['single_thread'] else 'No'} \\\\\n"
    #             first_timestamp = False
    #         else:
    #             latex_code += f"        & {date_dir} & {metrics['rmse']} & {metrics['after_opt_mean_psnr']} & {metrics['after_opt_mean_ssim']} & {'Yes' if metrics['single_thread'] else 'No'} \\\\\n"
    #     latex_code += "        \\hline\n"
    
# for i in range(3,5):
#     print("\n")
#     print(f"./ReplicaRendererCustom ../../datasets/office_{i}/mesh.ply ../../datasets/office_{i}/textures ../../datasets/office_{i}/glass.sur")
#     print(f"mkdir -p /datasets/replica_small/office{i}/results")
#     print(f"mv *.jpg /datasets/replica_small/office{i}/results")
#     print(f"mv *.png /datasets/replica_small/office{i}/results")
#     print(f"mv intrinsics.txt /datasets/replica_small/office{i}/")
#     print(f"cp /datasets/replica/office{i}/traj.txt /datasets/replica_small/office{i}/traj.txt")
# exit()        

# # Base directory for the results
# base_path = '/workspaces/src/MonoGS_dev/results/depth/replica_small'
# base_path = '/workspaces/src/MonoGS_dev/results/monocular/replica_small_cali'
base_path = '/workspaces/src/MonoGS_dev/results/monocular/replica_small'
output_csv_path = '/workspaces/src/MonoGS_dev/results/replica_small_cali_data_summary.csv'

# Collect data and write to CSV
data_dict, trj_dict = collect_data_from_directory(base_path)
latex_table_code = dict_to_rmse_latex(data_dict)
print(data_dict)
print(latex_table_code)
latex_table_code = dict_to_psnr_latex(data_dict)
print(latex_table_code)

# write_to_csv(data_dict, output_csv_path)
# plot_evo(trj_dict)
# print(f"Data collected and written to {output_csv_path}")

# result1_path = "/workspaces/src/MonoGS_dev/results/monocular/replica_cali/office4/2024-10-07-13-41-11"
# result2_path = "/workspaces/src/MonoGS_dev/results/monocular/replica_cali/office4/2024-10-07-17-55-24"

# trj_final_json_file_path1 = os.path.join(result1_path, 'plot', 'trj_final.json')
# trj_final_json_data1 = read_json_file(trj_final_json_file_path1)
# trj_dict1 = {}
# trj_data1 = dict()
# trj_data1['trj_id'] = trj_final_json_data1.get('trj_id', 'None')
# trj_data1['trj_est'] = trj_final_json_data1.get('trj_est', 'None')
# trj_data1['trj_gt']  = trj_final_json_data1.get('trj_gt', 'None')
# trj_dict1['office4']['2024-10-07-13-41-11'] = trj_data1

# trj_final_json_file_path2 = os.path.join(result2_path, 'plot', 'trj_final.json')
# trj_final_json_data2 = read_json_file(trj_final_json_file_path2)
# trj_dict2 = {}
# trj_data2 = dict()
# trj_data2['trj_id'] = trj_final_json_data2.get('trj_id', 'None')
# trj_data2['trj_est'] = trj_final_json_data2.get('trj_est', 'None')
# trj_data2['trj_gt']  = trj_final_json_data2.get('trj_gt', 'None')
# trj_dict2['office4']['2024-10-07-13-41-11'] = trj_data2
# plot_evo(trj_dict1)
