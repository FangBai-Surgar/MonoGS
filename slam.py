import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from slam_impl.config_utils import load_config
from slam_impl.dataset import load_dataset
from slam_impl.eval_utils import save_gaussians, save_cameras, save_ATE_records
from slam_impl.logging_utils import Log
from slam_impl.multiprocessing_utils import FakeQueue
from slam_impl.slam_backend import BackEnd
from slam_impl.slam_frontend import FrontEnd

from typing import NamedTuple

import random
import numpy as np


from gaussian_viewer import Viewer, create_gaussians_gl
from slam_impl.eval_cali_utils import eval_ate, eval_rendering, save_gaussians_class, save_cali


import shutil

def delete_files_and_subdirectories(directory_path):
   try:
     with os.scandir(directory_path) as entries:
       for entry in entries:
         if entry.is_file():
            os.unlink(entry.path)
         else:
            shutil.rmtree(entry.path)
     print("All files and subdirectories deleted successfully.")
   except OSError:
     print("Error occurred while deleting files and subdirectories.")



class OnlineCalibrationSettings:
    def __init__ (self):
        self.require_calibration = False
        self.allow_lens_distortion = False
        self.calib_module_test = False


class SLAM:
    def __init__(self, config, save_dir=None, calib_opts=OnlineCalibrationSettings()):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        # online calibration control
        self.frontend.MODULE_TEST_CALIBRATION = calib_opts.calib_module_test # for test and debugging, using --calib_module_test
        self.frontend.require_calibration = calib_opts.require_calibration
        self.frontend.allow_lens_distortion = calib_opts.allow_lens_distortion
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        # online calibration control
        self.backend.require_calibration = calib_opts.require_calibration
        self.backend.allow_lens_distortion = calib_opts.allow_lens_distortion
        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
            record_to_dir=os.path.join(save_dir, "recording") if self.config["Results"]["record"] else None
        )

        backend_process = mp.Process(target=self.backend.run)
        backend_process.name="SLAM Backend"
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.name="SLAM GUI"
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")
        save_gaussians(self.frontend.gaussians, self.save_dir, iteration="before_opt", final=False)
        save_cameras(self.save_dir, "final", self.frontend.cameras)
        save_ATE_records(self.save_dir, self.frontend.ATE_records)

        if self.eval_rendering:
            #clean frontend images
            for idx in self.frontend.cameras:
                self.frontend.cameras[idx].depth = None
            torch.cuda.empty_cache()

            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)
            # print the number of gaussians
            Log("Number of Gaussians: {}".format(self.gaussians.get_xyz.shape[0]))
            # save gaussians class
            save_gaussians_class(self.save_dir, self.gaussians)
            save_cali(self.save_dir, self.frontend.cameras, self.frontend.kf_indices, self.frontend.ATE_records)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass






if __name__ == "__main__":

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help='path to slam config file')
    parser.add_argument("--eval", action="store_true", default=False, help='default=False. evaluation mode which performs 3DGS refinement after completing SLAM')
    parser.add_argument("--calib_module_test", action="store_true", default=False, help='default=False. module test of selfcalibration')

    # options to overwrite config for test purpurses    
    parser.add_argument("--disable_selfcalibration", action="store_true", default=False, help='default=False. force to disable selfcalibration and overwrite config file') # overwrite config
    parser.add_argument("--disable_gui", action="store_true", default=False, help='default=False. force to disable gui') # overwrite config
    parser.add_argument("--use_gt_pose", action="store_true", default=False, help='default=False. use provided ground-truth poses') # use ground-truth poses
    parser.add_argument("--save_dir", type=str, default=None, help='path to save slam results, default=results')
    parser.add_argument("--record", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    calib_opts = OnlineCalibrationSettings()

    # selfcalibration control params in config
    if 'SelfCalibration' in config['Dataset'].keys():
        calib_opts.require_calibration = config['Dataset']['SelfCalibration']['enabled']
        calib_opts.allow_lens_distortion = True if config['Dataset']['SelfCalibration']['radial_distortion'] == 1 else False

    # test mode
    if args.calib_module_test:
        calib_opts.require_calibration = True
        calib_opts.allow_lens_distortion = True
        calib_opts.calib_module_test = True
        
    # disable selfcalibration
    if args.disable_selfcalibration:
        calib_opts.require_calibration = False
        calib_opts.allow_lens_distortion = False

    # disable gui
    if args.disable_gui:
        config["Results"]["use_gui"] = False

    if args.save_dir is not None:
        config["Results"]["save_dir"] = args.save_dir

    # use ground-truth poses
    if args.use_gt_pose:
        config["use_gt_pose"] = True
        config["Training"]["single_thread"] = True
        config["Dataset"]["single_thread"] = True

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True
        Log("\trequire_calibration=" + str(calib_opts.require_calibration))
        Log("\tallow_lens_distortion=" + str(calib_opts.allow_lens_distortion))
        Log("\tcalib_module_test=" + str(calib_opts.calib_module_test))


    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        # current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        base, name = os.path.split(args.config)
        head, tail = os.path.split(base)
        config_name = name[:-5]
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], tail, config_name
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        config["calib_opts_require_calibration"] = calib_opts.require_calibration
        config["calib_opts_allow_lens_distortion"] = calib_opts.allow_lens_distortion
        config["calib_module_test"] = calib_opts.calib_module_test
        delete_files_and_subdirectories(save_dir)
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{config_name}",
            config=config,
            mode="disabled",
            # mode=None if config["Results"]["use_wandb"] else "disabled"
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")
        config["Results"]["record"] = args.record



    slam = SLAM(config, save_dir=save_dir, calib_opts=calib_opts)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")