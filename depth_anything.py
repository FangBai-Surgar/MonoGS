

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

from numpy.polynomial import Polynomial, Chebyshev

import imgviz
import statistics
import operator

from submodules.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2




class DepthAnything:

    def __init__(self, encoder = 'vitb') -> None:
        # encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }

        self.model = DepthAnythingV2(**self.model_configs[encoder])
        self.model.load_state_dict(torch.load(f'submodules/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to('cuda').eval()

    def eval(self, raw_img):
        """
        Depth anything returns disparity:
            depth = (focal * baseline) / disparity
        https://github.com/LiheYoung/Depth-Anything/issues/47
        
        Parameters:
            raw_img:   np.array   H*W*3
        Returns:
            disparity: np.array   H*W
        """
        disparity = self.model.infer_image(raw_img) # HxW raw depth map in numpy
        return disparity
    
    @staticmethod
    # candidate colormaps: 'nipy_spectral', 'Spectral_r', 'turbo', 'plasma'
    def depth2image (depth, colormap=None):
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        if colormap is not None:
            cmap = matplotlib.colormaps.get_cmap(colormap)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        return depth

    @staticmethod
    def estimateScaleFactor (depth, uv_depth_stack):
        scale_stack = []
        for uv_d in  uv_depth_stack:
            x = int(round(uv_d[0]))
            y = int(round(uv_d[1]))
            s = uv_d[2] / depth[y, x]
            scale_stack.append(s)
        return statistics.median (scale_stack)


    @staticmethod
    def correct_depth_from_sparse_points (depth, uv_depth_stack):
        """
        Learn per depth correction function F(depth) - > scale,
        such that:
            depth * F(depth) = realDepth   <===>   F(depth) = realDepth / depth
        Parameters:
            a
        Returns:
            corrected_depth
        """
        depth_scale_stack = []
        for uv_d in  uv_depth_stack:
            x = int(round(uv_d[0]))
            y = int(round(uv_d[1]))
            d = depth[y, x] # depth
            s = uv_d[2] / d # realDepth / depth
            depth_scale_stack.append([d, s, uv_d[2]]) # (depth, scale, realDepth)

        depth_scale_stack.sort(key=operator.itemgetter(0)) # in-place sort, by depth
        depth_scale_array = np.array(depth_scale_stack).transpose()

        depth_array = depth_scale_array[0] # reference
        scale_array = depth_scale_array[1] # reference
        rldpt_array = depth_scale_array[2] # reference

        poly = np.polynomial.Chebyshev.fit (x=depth_array, y=scale_array, deg=3).convert(kind=np.polynomial.Polynomial)
        coef = poly.convert().coef

        tmp = depth
        corrected_depth = np.zeros_like(depth)

        for c in coef:
            corrected_depth = corrected_depth + c*tmp
            tmp = tmp * depth

        if True:
            print(f"check: error = {np.linalg.norm(scale_array - rldpt_array/depth_array)}")
            plt.rcParams["figure.figsize"] = (8,3)
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            ax1.plot(depth_array, 'r-', label="depth")
            ax1.plot(rldpt_array, 'b-', label="realDepth")
            ax1.set_xlabel("point IDs")
            ax1.set_ylabel("depth values")
            ax1.legend()
            ax2.plot(depth_array, scale_array, label="scale = realDepth/depth")
            ax2.set_xlabel("depth")
            ax2.set_ylabel('scale')
            ax2.legend()
            plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=0.0)
            plt.show()
            print(f"polynomial coefs = {coef}")
        
        return corrected_depth



if __name__ == "__main__":


    DA  = DepthAnything(encoder = 'vitl')

    image_dir = '/hdd/tandt/truck/images/000110.jpg'

    raw_img = cv2.imread(image_dir)

    depth = DA.eval(raw_img) # HxW raw depth map in numpy
    depth = DA.depth2image(depth, colormap='nipy_spectral')

    # depth = ( depth / np.median(depth) ) * 2.0
    # depth = imgviz.depth2rgb(depth, min_value=0.3, max_value=5.0)
    # depth = torch.from_numpy(depth)
    # depth = torch.permute(depth, (2, 0, 1)).float()
    # depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()

    print (f"rgb image_shape   :   {raw_img.shape}")
    print (f"depth image_shape :   {depth.shape}")


    cv2.imshow('raw-image', raw_img)
    cv2.imshow('depth-image', depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



