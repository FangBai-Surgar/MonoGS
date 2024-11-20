import os
# os.environ['MPLBACKEND'] = 'Agg'

# Base command
# types=['mono', 'rgbd']
types=['mono']
# data_dict = {
#     "0": ["_v6",''],
#     "1": ["_v0",''],
#     "2": ["_v3",''],
#     "3": ["_v9",'_v10',''],
#     "4": ["_v0",'']
# }
data_dict = {
    "0": ['_800600_300', '_800600_350', '_800600_400', '_800600_450', '_800600_500', '_800600_550', '_800600_600', '_800600_650', '_800600_700', '_800600_750', '_800600_800', '_800600_850']
}

for type in types:
    # for seq in seqs:
    for seq, versions in data_dict.items():
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
            # if os.path.exists(config_file_path):
            #     # mkdire output folder
            #     os.makedirs(f'./cmd_output/office{seq}', exist_ok=True)
            #     output_file = f'./cmd_output/office{seq}/office{seq}_{version}.txt'
            #     if dataset == 'replica_small_cali' and type == 'rgbd':
            #         base_command = f"python slam_cali.py --config {config_file_path} --eval --require_calibration | tee {output_file}"
            #         command = base_command.format(config_file=config_file, output_file=output_file)
                
            #     # Run the command
            #         print(f"Running: {command}")
            #         os.system(command)
            #         base_command = f"python slam_cali.py --config {config_file_path} --eval"
            #         command = base_command.format(config_file=config_file, output_file=output_file)
                
            #     # Run the command
            #         print(f"Running: {command}")
            #         os.system(command)
            #     if dataset == 'replica_small' and type == 'rgbd':
            #         base_command = f"python slam_cali.py --config {config_file_path} --eval"
            #         command = base_command.format(config_file=config_file, output_file=output_file)
                
            #     # Run the command
            #         print(f"Running: {command}")
            #         os.system(command)                
            #     if dataset == 'replica_small_cali' and type == 'mono':
            #         base_command = f"python slam_cali.py --config {config_file_path} --eval"
            #         command = base_command.format(config_file=config_file, output_file=output_file)
                
                # # Run the command
                #     print(f"Running: {command}")
                #     os.system(command)
                # base_command = f"python slam_cali.py --config {config_file_path} --eval --require_calibration --allow_lens_distortion | tee {output_file}"
            base_command = f"python slam_cali.py --config {config_file_path} --eval"
            # base_command = f"python slam_cali.py --config {config_file_path} --eval"
            
            # Construct the full command
            command = base_command.format(config_file=config_file)
            
            # # Run the command
            print(f"Running: {command}")
            os.system(command)

print("All done!")