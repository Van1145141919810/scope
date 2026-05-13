#!/usr/bin/env python
"""
Quick demo for group meeting — processes only first 3 samples.
Saves mask (ground truth) and prediction images to output/.
"""
import sys; sys.path.insert(0, 'scripts')
import torch
from model import scope, VaeTestDataset, set_seed, SEED1, IMG_SIZE, SEQ_LEN
from local_occ_grid_map import LocalMap
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

set_seed(SEED1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 3  # only 3 samples for quick demo
NUM_MC = 32

# Map params
MAP_X_LIMIT = [0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1
P_PRIOR = 0.5

def main():
    test_dir = sys.argv[1]
    mdl_path = sys.argv[2] if len(sys.argv) > 2 else "model/scope_model.pth"

    print(f"Device: {DEVICE}")
    print(f"Loading model from {mdl_path}...")

    model = scope(input_channels=1, latent_dim=512, output_channels=1).to(DEVICE)
    ckpt = torch.load(mdl_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']})")

    print(f"Loading test data from {test_dir}...")
    dataset = VaeTestDataset(test_dir, 'test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=NUM_SAMPLES)):
            if idx >= NUM_SAMPLES:
                break

            scans = batch['scan'].to(DEVICE)
            positions = batch['position'].to(DEVICE)
            velocities = batch['velocity'].to(DEVICE)
            batch_size = scans.size(0)

            # Build mask (ground truth future maps)
            mask_gridMap = LocalMap(MAP_X_LIMIT, MAP_Y_LIMIT, RESOLUTION, P_PRIOR,
                                     size=[batch_size, SEQ_LEN], device=DEVICE)
            x_odom = torch.zeros(batch_size, SEQ_LEN).to(DEVICE)
            y_odom = torch.zeros(batch_size, SEQ_LEN).to(DEVICE)
            theta_odom = torch.zeros(batch_size, SEQ_LEN).to(DEVICE)
            dist_future = scans[:, SEQ_LEN:]
            angles = torch.linspace(-135*np.pi/180, 135*np.pi/180, dist_future.shape[-1]).to(DEVICE)
            dx, dy = mask_gridMap.lidar_scan_xy(dist_future, angles, x_odom, y_odom, theta_odom)
            mask_maps = mask_gridMap.discretize(dx, dy).unsqueeze(2)

            # Build input maps (past frames)
            obs_pos_N = positions[:, SEQ_LEN - 1]
            vel_N = velocities[:, SEQ_LEN - 1]
            pos = positions[:, :SEQ_LEN]

            input_gridMap = LocalMap(MAP_X_LIMIT, MAP_Y_LIMIT, RESOLUTION, P_PRIOR,
                                      size=[batch_size, SEQ_LEN], device=DEVICE)
            pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T=1, noise_std=[0, 0, 0])
            x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)
            dist_past = scans[:, :SEQ_LEN]
            dx, dy = input_gridMap.lidar_scan_xy(dist_past, angles, x_odom, y_odom, theta_odom)
            input_maps = input_gridMap.discretize(dx, dy).unsqueeze(2)

            # Autoregressive 10-step prediction
            prediction_maps = torch.zeros(SEQ_LEN, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
            for step in range(SEQ_LEN):
                T = step + 1
                pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std=[0, 0, 0])
                x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)
                dx, dy = input_gridMap.lidar_scan_xy(dist_past, angles, x_odom, y_odom, theta_odom)
                input_binary = input_gridMap.discretize(dx, dy).unsqueeze(2)

                inputs_samples = input_binary.repeat(NUM_MC, 1, 1, 1, 1)
                for t in range(T):
                    pred, _ = model(inputs_samples)
                    pred = pred.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)
                    inputs_samples = torch.cat([inputs_samples[:, 1:], pred], dim=1)
                pred_mean = torch.mean(pred.squeeze(1), dim=0, keepdim=True)
                prediction_maps[step, 0] = pred_mean.squeeze()

            # Save ground truth
            fig, axes = plt.subplots(1, 10, figsize=(10, 1.2))
            for m in range(SEQ_LEN):
                axes[m].imshow(mask_maps[0, m, 0].cpu().numpy(), cmap='gray')
                axes[m].axis('off')
                axes[m].set_title(f"n={m+1}", fontsize=8)
            plt.tight_layout()
            fig.savefig(f"output/quick_mask{idx}.png", dpi=100)
            plt.close(fig)
            print(f"Saved output/quick_mask{idx}.png")

            # Save prediction
            fig, axes = plt.subplots(1, 10, figsize=(10, 1.2))
            for m in range(SEQ_LEN):
                axes[m].imshow(prediction_maps[m, 0].cpu().numpy(), cmap='gray')
                axes[m].axis('off')
                axes[m].set_title(f"n={m+1}", fontsize=8)
            plt.tight_layout()
            fig.savefig(f"output/quick_pred{idx}.png", dpi=100)
            plt.close(fig)
            print(f"Saved output/quick_pred{idx}.png")

    print("\n=== Done! Check output/quick_mask*.png and output/quick_pred*.png ===")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python quick_demo.py <test_dir> [model_path]")
        sys.exit(1)
    main()
