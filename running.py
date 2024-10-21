import os
os.environ['MPLBACKEND'] = 'Agg'

# Base command
# types=['mono', 'rgbd']
# seqs = ["0", "1", "2", "3", "4"]
# versions = ['', '_v0_sp', '_v0', '_v1_sp', '_v1', '_v2_sp', '_v2', '_v3_sp', '_v3', '_v4_sp', '_v4', '_v5_sp', '_v5', '_v6_sp', '_v6', '_v7', '_v8']


types=['mono']
seqs = ["0"]
versions = ['_v6']

# Loop through each version and run the command
for seq in seqs:
    for type in types:
        for version in versions:
            config_file = f'office{seq}{version}.yaml'
            # if there is _ in the version
            if 'v' in version:
                dataset = 'replica_small_cali'
            else:
                dataset = 'replica_small'
            # if there is config file
            config_file_path = f'./configs/{type}/{dataset}/{config_file}'
            print(f"config_file_path: {config_file_path}")
            if os.path.exists(config_file_path):
                # mkdire output folder
                os.makedirs(f'./cmd_output/office{seq}', exist_ok=True)
                output_file = f'./cmd_output/office{seq}/office{seq}_{version}.txt'
                base_command = f"python slam_cali.py --config {config_file_path} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
                
                # Construct the full command
                command = base_command.format(config_file=config_file, output_file=output_file)
                
                # Run the command
                print(f"Running: {command}")
                os.system(command)

print("All done!")