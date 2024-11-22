import os
from argparse import ArgumentParser
# python slam_cali.py --config /datasets/replica_small/office0_800600_500/office0_800x600_500.yaml --eval
# python run.py --config_base_dir /datasets/replica_small
# python run.py --config_base_dir ./configs/mono/replica_small


# write a function that iterative through the folders
def run_slam(dataset_path):
    for root, dirs, files in os.walk(dataset_path):
        # Sort directories and files to ensure a consistent order
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".yaml"):
                config_file_path = os.path.join(root, file)
                command = f"python slam_cali.py --config {config_file_path} --eval"
                # command = base_command.format(config_file=config_file, output_file=output_file)
                # print(f"Running: {command}")
                os.system(command)



if __name__ == "__main__":
    # add argparse to get the config file path
    parser = ArgumentParser(description="batch run slam calibration")
    parser.add_argument("--config_base_dir", type=str)
    args = parser.parse_args()
    run_slam(args.config_base_dir)
    