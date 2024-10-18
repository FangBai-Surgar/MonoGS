import os
os.environ['MPLBACKEND'] = 'Agg'
# Base command
types=['mono', 'rgbd']
seqs = ["0", "1", "2", "3", "4"]
versions = ['v0_sp', 'v0', 'v1_sp', 'v1', 'v2_sp', 'v2', 'v3_sp', 'v3', 'v4_sp', 'v4', 'v5_sp', 'v5', 'v6_sp', 'v6']
# versions = ['v5_sp', 'v5', 'v6_sp', 'v6']

# Loop through each version and run the command
for seq in seqs:
    for type in types:
        for version in versions:
            config_file = f'office{seq}_{version}.yaml'
            # if there is _ in the version
            if '_' in version:
                dataset = 'replica_small_cali'
            else:
                dataset = 'replica_small'
            # if there is config file
            if os.path.exists(f'./configs/{type}/{dataset}/{config_file}'):
                base_command = f"python slam_cali.py --config configs/{type}/{dataset}/{config_file} --eval --require_calibration --allow_lens_distortion | tee {output_file}"

                output_file = f'./cmd_output/office{seq}_{version}.txt'
                
                # Construct the full command
                command = base_command.format(config_file=config_file, output_file=output_file)
                
                # Run the command
                print(f"Running: {command}")
                os.system(command)

print("All done!")