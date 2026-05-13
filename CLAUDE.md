# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SCOPE (Stochastic Cartographic Occupancy Prediction Engine) — a deep learning framework for occupancy grid map prediction in dynamic environments, published at IEEE T-RO 2025. Three model variants on separate branches: **scope++** (highest accuracy, `scope++` branch), **scope** (balanced, `scope` branch), **so-scope** (fastest inference for resource-limited robots, `so-scope` branch). Current branch: `scope`.

Architecture: **ConvLSTM + β-VAE**. A ConvLSTM encodes a 10-frame sequence of local occupancy grid maps into a latent representation, a VAE encoder maps it to a Gaussian latent space, and a decoder reconstructs the predicted future occupancy map.

## Commands

```bash
# Activate environment
mamba activate scope

# Create/update environment
mamba env create -f environment.yaml

# Train (requires OGM-Datasets, see below)
sh run_train.sh <path_to_train_dir> <path_to_val_dir>

# Run inference demo with pre-trained model (requires OGM-Datasets)
sh run_eval_demo.sh <path_to_test_dir>

# Quick smoke test: load pre-trained model and run forward pass
python test_env.py
```

## Code architecture

All source code is in `scripts/`. No external package structure — imports rely on `sys.path` or running from `scripts/` directory.

### Core files

| File | Purpose |
|------|---------|
| `scripts/model.py` | **Main file.** Contains: (1) `scope` model class — ConvLSTM encoder → VAE encoder → reparameterization → decoder, (2) `VaeTestDataset` — reads `.npy` files of LiDAR scans/positions/velocities from dataset directories, (3) all architecture components (`Encoder`, `Decoder`, `Residual`, `ResidualStack`, `VAE_Encoder`), (4) global constants `SEQ_LEN=10`, `IMG_SIZE=64` |
| `scripts/convlstm.py` | Standalone ConvLSTM cell + multi-layer wrapper. Adapted from `ndrplz/ConvLSTM_pytorch`. Only `ConvLSTMCell` is actually used by the model |
| `scripts/local_occ_grid_map.py` | GPU-parallelized occupancy grid mapping. Uses Bresenham ray tracing to convert LiDAR scans to binary occupancy grids. Key class: `LocalMap` |
| `scripts/bresenham_torch.py` | N-dimensional Bresenham line algorithm implemented in PyTorch (GPU-compatible) |
| `scripts/train.py` | Training loop with β-VAE loss (`BCE + β*KL`). Uses `tensorboardX` for logging (not standard `tensorboard`). Loads data via `VaeTestDataset` from `model.py` |
| `scripts/decode_demo.py` | Multi-step inference demo. Builds occupancy grids from test data, performs autoregressive 10-step prediction with 32 Monte Carlo samples, outputs comparison images to `output/` |

### Data flow

1. LiDAR scans (1080-point) + positions + velocities loaded as `.npy` files via `VaeTestDataset`
2. `LocalMap` converts LiDAR data to 64×64 binary occupancy grids (past 10 and future 10 frames)
3. Past frames → ConvLSTM → VAE encoder → latent z ~ N(μ, σ)
4. Latent z → decoder → predicted occupancy grid (compared against future ground truth)
5. Loss: `BCELoss(prediction, future_grid) + 0.01 * KL_divergence`

### Key constants

- `SEQ_LEN = 10` (10 past frames as input, predict 10 future frames)
- `IMG_SIZE = 64` (64×64 grid maps)
- `POINTS = 1080` (LiDAR scan points)
- `NUM_LATENT_DIM = 512` (VAE latent dimension, reshaped to 2×16×16)
- `BETA = 0.01` (KL divergence weight in β-VAE)

### Pre-trained model

`model/scope_model.pth` — trained on OGM-Turtlebot2 dataset (epoch 40). 730K parameters.

## Dataset

**OGM-Datasets** from [Zenodo](https://doi.org/10.5281/zenodo.7051560). Three subsets:

- **OGM-Turtlebot2** — simulated Turtlebot2 in Gazebo lobby with 34 moving pedestrians
- **OGM-Jackal** — real Jackal robot, outdoor UT Austin (from SCAND dataset)
- **OGM-Spot** — real Spot robot, UT Austin Union Building (from SCAND dataset)

Data structure per subset: `{train,val,test}/scans/`, `positions/`, `velocities/` directories with `.npy` files + a `.txt` file listing them per split.

## Environment

Created via `environment.yaml`. Original paper used Python 3.7 + PyTorch 1.7.1, but this env uses **Python 3.9 + PyTorch 2.7.1** for CUDA 12/13 compatibility. The code uses only standard PyTorch APIs, no version issues.

Shell scripts (`run_train.sh`, `run_eval_demo.sh`) use Unix paths. On Windows, run from Git Bash or use `mamba run -n scope python scripts/...` directly.
