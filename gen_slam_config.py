import os
import yaml
from collections import OrderedDict
import argparse
import sys


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

def config_OrderedDict(inherit_from = 'configs/mono/replica_small/base_config.yaml',
                       dataset_path = '/datasets/replica_small/office0',
                       width = 680,
                       height = 480,
                       fx = 500,
                       fy = None,
                       cx = None,
                       cy = None,
                       k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0,
                       depth_scale = 6553.5,
                       distorted = False,
                       selfcalib_enabled = True,
                       selfcalib_radial = 0,
                       selfcalib_frame_id = None,
                       selfcalib_gt_fx = None,
                       backend_params_lr_cnt1 = 0.002,
                       backend_params_lr_cnt2 = 0.002,
                       grad_mask_row = 32,
                       grad_mask_col = 32,
                       single_thread = False,
                       dataset_type = 'replica'):

    fy = fx if fy is None else fy
    cx = ( width - 1)*0.5 if cx is None else cx
    cy = (height - 1)*0.5 if cy is None else cy

    config = OrderedDict([
            ('inherit_from', inherit_from),
            ('Dataset', OrderedDict([
                    ('dataset_path', dataset_path),
                    ('type', dataset_type),
                    ('single_thread', single_thread),
                    ('Calibration', OrderedDict([
                            ('fx', fx),
                            ('fy', fy),
                            ('cx', cx),
                            ('cy', cy),
                            ('k1', k1),
                            ('k2', k2),
                            ('p1', p1),
                            ('p2', p2),
                            ('k3', k3),
                            ('width', width),
                            ('height', height),
                            ('depth_scale', depth_scale),
                            ('distorted', distorted)
                            ])),
                    ('SelfCalibration', OrderedDict([
                            ('enabled', selfcalib_enabled),
                            ('radial_distortion', selfcalib_radial),
                            ('frame_id', selfcalib_frame_id),
                            ('gt_fx', selfcalib_gt_fx),
                            ('backend_params', OrderedDict([
                                        ('lr_cnt1', backend_params_lr_cnt1),
                                        ('lr_cnt2', backend_params_lr_cnt2)
                                        ]))
                            ])),
                    ('grad_mask_row', grad_mask_row),
                    ('grad_mask_col', grad_mask_col)
                ]))
        ])
    
    return config


def test():

    config = config_OrderedDict(inherit_from = 'configs/mono/replica_small/base_config.yaml',
                                dataset_path = '/datasets/replica_small/office0',
                                width = 680,
                                height = 480,
                                fx = 500,
                                selfcalib_frame_id = "100, 200, 300",
                                selfcalib_gt_fx= "400, 300, 200")
    print(config)

    yaml_file_path = "test.yaml"
    with open(yaml_file_path, 'w') as file:
        ordered_dump(config, file, Dumper=yaml.SafeDumper, default_flow_style=False)

    config_loaded = ordered_load(yaml_file_path, Loader=yaml.Loader, object_pairs_hook=OrderedDict)
    print(config_loaded)


if __name__ == "__main__":
 
    # test()

    # python gen_slam_config.py --yaml_file_path "slam_config_example.yaml" --fx 100 --selfcalib_frame_id "100, 200, 300, 400, 500"

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description='SLAM YAML File Generator',
                        epilog='Text at the bottom of help')
    

    parser.add_argument('--inherit_from', type=str, default='configs/mono/replica_small/base_config.yaml')
    parser.add_argument('--dataset_path', type=str, default='/datasets/replica_small/office0')
    parser.add_argument('--width', type=int, default=800)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--fx', type=float, default=500)
    parser.add_argument('--fy', type=float, default=None)
    parser.add_argument('--selfcalib_frame_id', type=str, default=None)
    parser.add_argument('--selfcalib_gt_fx', type=str, default=None)
    parser.add_argument('--yaml_file_path', type=str, default="test.yaml")

    args = parser.parse_args(sys.argv[1:])

    config = config_OrderedDict(inherit_from = args.inherit_from,
                                dataset_path = args.dataset_path,
                                width = args.width,
                                height = args.height,
                                fx = args.fx,
                                fy = args.fy,
                                selfcalib_frame_id = args.selfcalib_frame_id,
                                selfcalib_gt_fx= args.selfcalib_gt_fx
                                )

    yaml_file_path = args.yaml_file_path
    with open(yaml_file_path, 'w') as file:
        ordered_dump(config, file, Dumper=yaml.SafeDumper, default_flow_style=False)
