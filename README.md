[comment]: <> (# Gaussian Splatting SLAM and Camera Instrinsic Calibration)


# Getting Started

## Environment setup
- Ubuntu 22.04
- nvidia-driver-535
- cuda-toolkit 11.8
- Python 3.9


Dependencies
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

pip install numpy matplotlib rerun-sdk
pip install opencv-contrib-python
pip install opencv-python
pip install pyyaml
pip install munch
pip install wandb
pip install open3d
pip install glfw
pip install imgviz
pip install pyopengl
pip install PyGLM
pip install rich
pip install trimesh
pip install evo
pip install -U torchmetrics
pip install plyfile tqdm
pip install evo
pip install gradio_imageslider
pip install gradio==4.29.0
pip install pycolmap
pip install imgui
pip install imageio
```

## Installation
```
git clone git@github.com:FangBai-Surgar/MonoGS.git --recursive
cd MonoGS
```

Setup
```
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```


## Quick Demo
```
bash scripts/download_tum.sh
python slam.py --config configs/mono/tum/fr3_office.yaml
```
You will see a GUI window pops up.

## Downloading Datasets
Running the following scripts will automatically download datasets to the `./datasets` folder.
### TUM-RGBD dataset
```bash
bash scripts/download_tum.sh
```

### Replica dataset
```bash
bash scripts/download_replica.sh
```

### EuRoC MAV dataset
```bash
bash scripts/download_euroc.sh
```



## Run
### Monocular
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

### RGB-D
```bash
python slam.py --config configs/rgbd/tum/fr3_office.yaml
```

```bash
python slam.py --config configs/rgbd/replica/office0.yaml
```
Or the single process version as
```bash
python slam.py --config configs/rgbd/replica/office0_sp.yaml
```


### Stereo (experimental)
```bash
python slam.py --config configs/stereo/euroc/mh02.yaml
```

## Live demo with Realsense
First, you'll need to install `pyrealsense2`.
Inside the conda environment, run:
```bash
pip install pyrealsense2
```
Connect the realsense camera to the PC on a **USB-3** port and then run:
```bash
python slam.py --config configs/live/realsense.yaml
```
We tested the method with [Intel Realsense d455](https://www.mouser.co.uk/new/intel/intel-realsense-depth-camera-d455/). We recommend using a similar global shutter camera for robust camera tracking. Please avoid aggressive camera motion, especially before the initial BA is performed. Check out [the first 15 seconds of our YouTube video](https://youtu.be/x604ghp9R_Q?si=S21HgeVTVfNe0BVL) to see how you should move the camera for initialisation. We recommend to use the code in `dev.speed-up` branch for live demo.

<p align="center">
  <a href="">
    <img src="./media/realsense.png" alt="teaser" width="50%">
  </a>
</p>

# Evaluation
<!-- To evaluate the method, please run the SLAM system with `save_results=True` in the base config file. This setting automatically outputs evaluation metrics in wandb and exports log files locally in save_dir. For benchmarking purposes, it is recommended to disable the GUI by setting `use_gui=False` in order to maximise GPU utilisation. For evaluating rendering quality, please set the `eval_rendering=True` flag in the configuration file. -->
To evaluate our method, please add `--eval` to the command line argument:
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```
This flag will automatically run our system in a headless mode, and log the results including the rendering metrics.

# Reproducibility
There might be minor differences between the released version and the results in the paper. Please bear in mind that multi-process performance has some randomness due to GPU utilisation.
We run all our experiments on an RTX 4090, and the performance may differ when running with a different GPU.

# Acknowledgement
This work incorporates many open-source codes. We extend our gratitude to the authors of the software.
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Differential Gaussian Rasterization
](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [SIBR_viewers](https://gitlab.inria.fr/sibr/sibr_core)
- [Tiny Gaussian Splatting Viewer](https://github.com/limacv/GaussianSplattingViewer)
- [Open3D](https://github.com/isl-org/Open3D)
- [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

# License
MonoGS is released under a **LICENSE.md**. For a list of code dependencies which are not property of the authors of MonoGS, please check **Dependencies.md**.

# Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```bibtex
@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

```













