import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.slam_utils import image_gradient, image_gradient_mask
from utils.config_utils import load_config
from utils_cali.dataset_cali import load_dataset
from munch import munchify
from utils_cali.camera_cali_utils import CameraForCalibration as Camera
import cv2


# visualize the gradient mask
def update(replica_origin_dataset, replica_cali_dataset, replica_cali_16_45_dataset, cur_frame_idx):
    viewpoint_origin = Camera.init_from_dataset(replica_origin_dataset, cur_frame_idx)
    replica_origin_img = viewpoint_origin.original_image.cuda().cpu().numpy().transpose(1, 2, 0)  # Transpose to H, W, C
    viewpoint_origin.compute_grad_mask(replica_config)
    origin_grad_mask = viewpoint_origin.grad_mask.cpu().numpy()[0]  # Assuming single-channel mask, take the first slice
    origin_grad_mask = np.repeat(origin_grad_mask[:, :, np.newaxis], 3, axis=2)  # Repeat along the third axis to make it 3-channel

    viewpoint_cali = Camera.init_from_dataset(replica_cali_dataset, cur_frame_idx)
    replica_cali_img = viewpoint_cali.original_image.cuda().cpu().numpy().transpose(1, 2, 0)  # Transpose to H, W, C
    viewpoint_cali.compute_grad_mask(replica_cali_config)
    cali_grad_mask = viewpoint_cali.grad_mask.cpu().numpy()[0]  # Assuming single-channel mask, take the first slice
    cali_grad_mask = np.repeat(cali_grad_mask[:, :, np.newaxis], 3, axis=2)  # Make 3-channel

    viewpoint_cali_16_45 = Camera.init_from_dataset(replica_cali_16_45_dataset, cur_frame_idx)
    viewpoint_cali_16_45_img = viewpoint_cali_16_45.original_image.cuda().cpu().numpy().transpose(1, 2, 0)  # Transpose to H, W, C
    viewpoint_cali_16_45.compute_grad_mask(replica_cali_config_16_45)
    cali_grad_mask_16_45 = viewpoint_cali_16_45.grad_mask.cpu().numpy()[0]  # Assuming single-channel mask, take the first slice
    cali_grad_mask_16_45 = np.repeat(cali_grad_mask_16_45[:, :, np.newaxis], 3, axis=2)  # Make 3-channel


    # Resize both calibrated image and its gradient mask to match the origin image dimensions
    replica_cali_img = cv2.resize(replica_cali_img, (replica_origin_img.shape[1], replica_origin_img.shape[0]))
    cali_grad_mask = cv2.resize(cali_grad_mask, (replica_origin_img.shape[1], replica_origin_img.shape[0]))
    replica_cali_16_45_img = cv2.resize(replica_cali_img, (replica_origin_img.shape[1], replica_origin_img.shape[0]))
    cali_grad_mask_16_45 = cv2.resize(cali_grad_mask_16_45, (replica_origin_img.shape[1], replica_origin_img.shape[0]))

    return replica_origin_img, origin_grad_mask, replica_cali_img, cali_grad_mask, replica_cali_16_45_img, cali_grad_mask_16_45


replica_config_path = "./configs/mono/replica_small/office3_sp.yaml"
replica_cali_config_path = "./configs/mono/replica_small_cali/office3_v5_sp.yaml"


replica_config = load_config(replica_config_path)
model_params_origin = munchify(replica_config["model_params"])
replica_origin_dataset = load_dataset(model_params_origin, model_params_origin.source_path, config=replica_config)

replica_cali_config = load_config(replica_cali_config_path)
model_params = munchify(replica_cali_config["model_params"])
replica_cali_dataset = load_dataset(model_params, model_params.source_path, config=replica_cali_config)

replica_cali_config_16_45 = load_config(replica_cali_config_path)
replica_cali_config_16_45["Dataset"]["grad_mask_row"] = 32
replica_cali_config_16_45["Dataset"]["grad_mask_col"] = 32
replica_cali_config_16_45["Training"]["edge_threshold"] = 3.2
replica_cali_16_45_dataset = load_dataset(model_params, model_params.source_path, config=replica_cali_config_16_45)



cur_frame_idx = 0



    
images = update(replica_origin_dataset, replica_cali_dataset, replica_cali_16_45_dataset, cur_frame_idx)
print(images[0].shape, images[1].shape, images[2].shape, images[3].shape)
# Window setup

cv2.namedWindow('Image Viewer', cv2.WINDOW_NORMAL)

while True:
    # Display images in a single window
    combined_img = np.vstack([
        np.hstack([images[0], images[1]]),  # Top row: original image and its gradient mask
        np.hstack([images[3], images[2]]),   # Bottom row: calibrated image and its gradient mask
        np.hstack([images[4], images[5]])   # Bottom row: calibrated image and its gradient mask
    ])
    cv2.imshow('Image Viewer', combined_img)
    
    # Wait for key press
    key = cv2.waitKey(0)
    print(f"Key pressed: {key}")
    
    if key == 27:  # ESC key to exit
        break
    elif key == 83:  # Right arrow key (key code may vary; check cv2 documentation)
        cur_frame_idx += 1  # Move to the next frame
        images = update(replica_origin_dataset, replica_cali_dataset, replica_cali_16_45_dataset, cur_frame_idx)
        print(cur_frame_idx)
    elif key == 81:  # Left arrow key (key code may vary; check cv2 documentation)
        cur_frame_idx -= 1 if cur_frame_idx >1 else 0 # Move to the previous frame
        print(cur_frame_idx)
        images = update(replica_origin_dataset, replica_cali_dataset, replica_cali_16_45_dataset, cur_frame_idx)
        if cur_frame_idx >= 600:
            cur_frame_idx = 0  # Reset to first frame if we exceed the range

cv2.destroyAllWindows()



# simulated_img = np.array(Image.open(simulated_img_path))
# simulated = (
#     torch.from_numpy(simulated_img / 255.0)
#     .clamp(0.0, 1.0)
#     .permute(2, 0, 1)
#     .to(device="cuda:0", dtype=torch.float32)
# )
# # for simulated dataset, we need to provide the edge threshold for testing
# fig, axs = plt.subplots(4, 4)
# for i in range(1, 9):
#     # th = 2-4
#     th = 2 + (4-2)/8*(i-1)
#     simulated_grad_mask = compute_grad_mask(simulated, simulated_config, edg_th=th)
#     axs[(i-1) // 2, i%2].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
#     axs[(i-1) // 2, i%2].set_title(f"Grad Mask (edge th = {th})")

# for i in range(0, 4):
#     th = 2 + (4-2)/8*i
#     pa_size = 8
#     simulated_grad_mask = compute_grad_mask(simulated, replica_config, edg_th=th, patch_size=pa_size)
#     axs[i, 2].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
#     axs[i, 2].set_title(f"Patch {pa_size} Grad Mask (edge th = {th})")

# for i in range(0, 4):
#     th = 2 + (4-2)/8*i
#     pa_size = 64
#     simulated_grad_mask = compute_grad_mask(simulated, replica_config, edg_th=th, patch_size=pa_size)
#     axs[i, 3].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
#     axs[i, 3].set_title(f"Patch {pa_size} Grad Mask (edge th = {th})")
# plt.show()
# simulated_grad_mask = compute_grad_mask(simulated, simulated_config, edg_th=0.5)
# # visualize the image and the gradient mask

# # # figure 1
# # fig, axs = plt.subplots(1, 2)
# # axs[0].imshow(tum_img)
# # axs[0].set_title("TUM RGB Image")
# # axs[0].axis("off")
# # axs[1].imshow(tum_grad_mask.cpu().numpy().squeeze(), cmap="gray")
# # axs[1].set_title("TUM Gradient Mask")
# # # axs[1].axis("off")
# # plt.show()
# # figure 2
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(replica_img)
# axs[0].set_title("Replica RGB Image")
# axs[0].axis("off")
# axs[1].imshow(replica_grad_mask.cpu().numpy().squeeze(), cmap="gray")
# axs[1].set_title("Replica Gradient Mask")
# # axs[1].axis("off")
# plt.show()
# # # figure 3
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(simulated_img)
# axs[0].set_title("Simulated RGB Image")
# axs[0].axis("off")
# axs[1].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
# axs[1].set_title("Simulated Gradient Mask")
# axs[1].axis("off")
# plt.show()
