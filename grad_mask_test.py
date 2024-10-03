import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.slam_utils import image_gradient, image_gradient_mask
from utils.config_utils import load_config



# visualize the gradient mask
def compute_grad_mask(original_image, config, edg_th=None, patch_size=32):
    edge_threshold = config["Training"]["edge_threshold"]

    gray_img = original_image.mean(dim=0, keepdim=True)
    gray_grad_v, gray_grad_h = image_gradient(gray_img)
    mask_v, mask_h = image_gradient_mask(gray_img)
    gray_grad_v = gray_grad_v * mask_v
    gray_grad_h = gray_grad_h * mask_h
    img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

    if config["Dataset"]["type"] == "replica":
        row, col = patch_size, patch_size
        edge_threshold = edg_th if edg_th is not None else edge_threshold
        multiplier = edge_threshold
        _, h, w = original_image.shape
        for r in range(row):
            for c in range(col):
                block = img_grad_intensity[
                    :,
                    r * int(h / row) : (r + 1) * int(h / row),
                    c * int(w / col) : (c + 1) * int(w / col),
                ]
                th_median = block.median()
                block[block > (th_median * multiplier)] = 1
                block[block <= (th_median * multiplier)] = 0
        print(int(h / row))
        grad_mask = img_grad_intensity
    elif config["Dataset"]["type"] == "simulated":
        edg_th = edg_th if edg_th is not None else edge_threshold
        print(f"Edge threshold: {edg_th}")
        median_img_grad_intensity = img_grad_intensity.median()
        grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edg_th
        )
    else:
        median_img_grad_intensity = img_grad_intensity.median()
        grad_mask = (
            img_grad_intensity > median_img_grad_intensity * edge_threshold
        )
    return grad_mask

tum_img_path = "/datasets/tum/rgbd_dataset_freiburg3_long_office_household/rgb/1341847980.722988.png"
tum_fr3_config_path = "./configs/rgbd/tum/fr3_office.yaml"

simulated_img_path = "/datasets/mono-cali/image_sequence_1/FTF_00040.png"
simulated_config_path = "./configs/mono/simulated/seq1.yaml"

replica_img_path = "/datasets/replica/office0/results/frame000492.jpg"
replica_config_path = "./configs/rgbd/replica/office0.yaml"

tum_config = load_config(tum_fr3_config_path)
replica_config = load_config(replica_config_path)
simulated_config = load_config(simulated_config_path)


tum_img = np.array(Image.open(tum_img_path))
tum = (
    torch.from_numpy(tum_img / 255.0)
    .clamp(0.0, 1.0)
    .permute(2, 0, 1)
    .to(device="cuda:0", dtype=torch.float32)
)
tum_grad_mask = compute_grad_mask(tum, tum_config)

replica_img = np.array(Image.open(replica_img_path))
replica = (
    torch.from_numpy(replica_img / 255.0)
    .clamp(0.0, 1.0)
    .permute(2, 0, 1)
    .to(device="cuda:0", dtype=torch.float32)
)
replica_grad_mask = compute_grad_mask(replica, replica_config)

simulated_img = np.array(Image.open(simulated_img_path))
simulated = (
    torch.from_numpy(simulated_img / 255.0)
    .clamp(0.0, 1.0)
    .permute(2, 0, 1)
    .to(device="cuda:0", dtype=torch.float32)
)
# for simulated dataset, we need to provide the edge threshold for testing
fig, axs = plt.subplots(4, 4)
for i in range(1, 9):
    # th = 2-4
    th = 2 + (4-2)/8*(i-1)
    simulated_grad_mask = compute_grad_mask(simulated, simulated_config, edg_th=th)
    axs[(i-1) // 2, i%2].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
    axs[(i-1) // 2, i%2].set_title(f"Grad Mask (edge th = {th})")

for i in range(0, 4):
    th = 2 + (4-2)/8*i
    pa_size = 8
    simulated_grad_mask = compute_grad_mask(simulated, replica_config, edg_th=th, patch_size=pa_size)
    axs[i, 2].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
    axs[i, 2].set_title(f"Patch {pa_size} Grad Mask (edge th = {th})")

for i in range(0, 4):
    th = 2 + (4-2)/8*i
    pa_size = 64
    simulated_grad_mask = compute_grad_mask(simulated, replica_config, edg_th=th, patch_size=pa_size)
    axs[i, 3].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
    axs[i, 3].set_title(f"Patch {pa_size} Grad Mask (edge th = {th})")
plt.show()
simulated_grad_mask = compute_grad_mask(simulated, simulated_config, edg_th=0.5)
# visualize the image and the gradient mask

# # figure 1
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(tum_img)
# axs[0].set_title("TUM RGB Image")
# axs[0].axis("off")
# axs[1].imshow(tum_grad_mask.cpu().numpy().squeeze(), cmap="gray")
# axs[1].set_title("TUM Gradient Mask")
# # axs[1].axis("off")
# plt.show()
# figure 2
fig, axs = plt.subplots(1, 2)
axs[0].imshow(replica_img)
axs[0].set_title("Replica RGB Image")
axs[0].axis("off")
axs[1].imshow(replica_grad_mask.cpu().numpy().squeeze(), cmap="gray")
axs[1].set_title("Replica Gradient Mask")
# axs[1].axis("off")
plt.show()
# # figure 3
fig, axs = plt.subplots(1, 2)
axs[0].imshow(simulated_img)
axs[0].set_title("Simulated RGB Image")
axs[0].axis("off")
axs[1].imshow(simulated_grad_mask.cpu().numpy().squeeze(), cmap="gray")
axs[1].set_title("Simulated Gradient Mask")
axs[1].axis("off")
plt.show()
