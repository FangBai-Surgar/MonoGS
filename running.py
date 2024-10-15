import os

# Base command
base_command = "python slam_cali.py --config configs/mono/replica_small_cali/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
versions = ['v0_sp', 'v0', 'v1_sp', 'v1', 'v2_sp', 'v2', 'v3_sp', 'v3', 'v4_sp', 'v4', 'v5_sp', 'v5', 'v6_sp', 'v6']
# versions = ['v5_sp', 'v5', 'v6_sp', 'v6']

# Loop through each version and run the command
for version in versions:
    config_file = f'office0_{version}.yaml'
    output_file = f'./cmd_output/office0_{version}.txt'
    
    # Construct the full command
    command = base_command.format(config_file=config_file, output_file=output_file)
    
    # Run the command
    print(f"Running: {command}")
    os.system(command)

base_command = "python slam_cali.py --config configs/mono/replica_small_cali/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
versions = ['v0_sp', 'v0', 'v1_sp', 'v1', 'v2_sp', 'v2']
# versions = ['v5_sp', 'v5', 'v6_sp', 'v6']

# Loop through each version and run the command
for version in versions:
    config_file = f'office2_{version}.yaml'
    output_file = f'./cmd_output/office2_{version}.txt'
    
    # Construct the full command
    command = base_command.format(config_file=config_file, output_file=output_file)

# base_command = "python slam_cali.py --config configs/mono/replica_small/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
# versions = ['', '_sp']
# # versions = ['_sp']
# # for i in range(2, 5):
# for i in range(1,2):
#     for version in versions:
#         config_file = f'office{i}{version}.yaml'
#         output_file = f'./cmd_output/office{i}{version}.txt'
        
#         # Construct the full command
#         command = base_command.format(config_file=config_file, output_file=output_file)
        
#         # Run the command
#         print(f"Running: {command}")
#         os.system(command)

# base_command = "python slam_cali.py --config configs/rgbd/replica_small/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
# versions = ['', '_sp']
# # versions = ['_sp']
# # for i in range(2, 5):
# for i in range(0,5):
#     for version in versions:
#         config_file = f'office{i}{version}.yaml'
#         output_file = f'./cmd_output/office{i}{version}_rgbd.txt'
        
#         # Construct the full command
#         command = base_command.format(config_file=config_file, output_file=output_file)
        
#         # Run the command
#         print(f"Running: {command}")
#         os.system(command)

base_command = "python slam_cali.py --config configs/rgbd/replica_small_cali/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
versions = ['v0_sp', 'v0', 'v1_sp', 'v1', 'v2_sp', 'v2', 'v3_sp', 'v3', 'v4_sp', 'v4', 'v5_sp', 'v5', 'v6_sp', 'v6']
# versions = ['v5_sp', 'v5', 'v6_sp', 'v6']

# Loop through each version and run the command
for version in versions:
    config_file = f'office0_{version}.yaml'
    output_file = f'./cmd_output/office0_{version}_rgbd.txt'
    
    # Construct the full command
    command = base_command.format(config_file=config_file, output_file=output_file)
    
    # Run the command
    print(f"Running: {command}")
    os.system(command)


base_command = "python slam_cali.py --config configs/rgbd/replica_small_cali/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
versions = ['v0_sp', 'v0', 'v1_sp', 'v1', 'v2_sp', 'v2']
# versions = ['v5_sp', 'v5', 'v6_sp', 'v6']

# Loop through each version and run the command
for version in versions:
    config_file = f'office2_{version}.yaml'
    output_file = f'./cmd_output/office2_{version}_rgbd.txt'
    
    # Construct the full command
    command = base_command.format(config_file=config_file, output_file=output_file)
    
    # Run the command
    print(f"Running: {command}")
    os.system(command)
