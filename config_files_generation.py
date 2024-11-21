import os
import yaml
from collections import OrderedDict

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

# Usage in your script
def generate_config_files(base_dir, base_file, offices, widths, heights, focals):
    base_path = os.path.join(base_dir, base_file)
    with open(base_path, 'r') as file:
        base_config = ordered_load(file, yaml.SafeLoader)
    
    data_dict = {}

    # Loop through the specified office numbers, dimensions, and focal lengths
    for office in offices:
        office_key = f"{office}"  # Create a dynamic key for each office
        data_dict[office_key] = []  # Initialize the list for this office
        
        for width in widths:
            for height in heights:
                for focal in focals:
                    # Modify the base config for each dimension and focal length
                    config = base_config.copy()
                    config['Dataset']['dataset_path'] = f"/datasets/replica_small/office{office}_{width}{height}_{focal}"
                    config['Dataset']['Calibration']['fx'] = focal
                    config['Dataset']['Calibration']['fy'] = focal
                    config['Dataset']['Calibration']['cx'] = width / 2 - 0.5
                    config['Dataset']['Calibration']['cy'] = height / 2 - 0.5
                    config['Dataset']['Calibration']['width'] = width
                    config['Dataset']['Calibration']['height'] = height

                    # Create new file name based on parameters
                    new_filename = f"office{office}_{width}{height}_{focal}.yaml"
                    new_file_path = os.path.join(base_dir, new_filename)

                    data_dict[office_key].append(f"_{width}{height}_{focal}")

                    # Write the modified configuration to a new YAML file
                    with open(new_file_path, 'w') as file:
                        ordered_dump(config, file, Dumper=yaml.SafeDumper, default_flow_style=False)

                    print(f"Generated config file: {new_file_path}")
    
    print("\n\ndata_dict = {")
    for key, values in data_dict.items():
        # Sort values and remove duplicates
        unique_sorted_values = sorted(set(values), key=lambda x: int(x.split('_')[-1]))
        formatted_values = ", ".join(f"'{value}'" for value in unique_sorted_values)
        print(f"    \"{key}\": [{formatted_values}]")
    print("}")
# print the data_dict

print("Generating configuration files...")

# Configuration settings
base_directory = "/workspaces/src/MonoGS_dev/configs/mono/replica_small"
# base_directory = "/workspaces/src/MonoGS_dev/configs/mono/replica_small_cali"
base_filename = "example.yaml"
office_numbers = range(0, 1)  # Generate for office0 to office4
widths = [800]           # Example widths
heights = [600]          # Example heights
focal_lengths = [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]  # Example focal lengths

# Generate the configuration files
generate_config_files(base_directory, base_filename, office_numbers, widths, heights, focal_lengths)
