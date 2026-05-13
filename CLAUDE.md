# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SCOPE (Stochastic Cartographic Occupancy Prediction Engine) — deep learning for occupancy grid map prediction in dynamic environments, IEEE T-RO 2025. Three model variants on separate branches: **scope++** (highest accuracy), **scope** (balanced, current branch), **so-scope** (fastest inference).

Architecture: **ConvLSTM + β-VAE**. A ConvLSTM encodes 10 frames of local occupancy grids → VAE encoder maps to Gaussian latent space → decoder reconstructs the predicted future occupancy map. Trained with `loss = BCE(pred, gt) + 0.01 × KL(q||p)`.

## Commands

```bash
# Activate environment
mamba activate scope

# Create/update environment from yaml
mamba env create -f environment.yaml

# Quick smoke test: verify env + model loading
python test_env.py

# Fast demo: 3 samples, ~25 seconds (for presentations)
python quick_demo.py ~/data/OGM-datasets/OGM-Turtlebot2/test model/scope_model.pth

# Train (requires OGM-Datasets; pre-trained model already exists)
sh run_train.sh <path_to_train_dir> <path_to_val_dir>

# Full inference demo on test set (very slow — 17k samples; use quick_demo.py instead)
sh run_eval_demo.sh <path_to_test_dir>
```

On Windows, shell scripts need Git Bash. Alternative: `mamba run -n scope python scripts/<script>.py <args>` from project root.

## Code architecture

All source in `scripts/`. Imports rely on `sys.path` or running from project root (scripts import from each other as flat modules). Every file now has detailed Chinese inline comments explaining each line.

### Core files and data flow

```
bresenham_torch.py          ← GPU Bresenham line algorithm (lowest-level utility)
    ↓
local_occ_grid_map.py       ← LiDAR → occupancy grid mapping (uses Bresenham for free space)
    ↓
convlstm.py                 ← ConvLSTM cell (spatial LSTM: convolutions replace matrix multiplies)
    ↓
model.py                    ← ★ Heart of the project:
                               (1) VaeTestDataset — loads .npy sequences of LiDAR/pose/velocity
                               (2) Residual/Encoder/Decoder — VAE building blocks
                               (3) scope model — ConvLSTM → VAE_Encoder → reparameterize → Decoder
    ↓
train.py                    ← Training loop: β-VAE loss, supervised only on future frame 1
    ↓
decode_demo.py              ← Inference: autoregressive 10-step prediction with 32 MC samples
```

### Key architectural decisions

- **Why log-odds for grid maps?** Probabilities can't be added directly. Log-odds can: `log_odds(p) = ln(p/(1-p))`. Prior p=0.5 → log_odds=0. Free space shifts negative, occupied shifts positive.
- **Why reparameterization trick?** Sampling `z ~ N(μ,σ²)` is non-differentiable. Rewrite as `z = μ + ε·σ, ε~N(0,1)` — gradient flows through μ and σ.
- **Why KL divergence in VAE?** Regularizes latent space toward N(0,1), making it smooth and continuous. Without it, the model degenerates to a standard autoencoder.
- **Why β=0.01 (small KL weight)?** Prioritizes reconstruction accuracy over latent regularity. The occupancy prediction task needs precise spatial output.
- **Why coordinate transform?** The robot moves during the 10-frame window. All past observations must be aligned to the predicted future reference frame before building grid maps.

### Key constants (defined in model.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `SEQ_LEN` | 10 | 10 past frames input, 10 future frames predicted |
| `IMG_SIZE` | 64 | 64×64 grid map |
| `POINTS` | 1080 | LiDAR scan points per frame |
| `NUM_LATENT_DIM` | 512 | VAE latent dim (= 2 channels × 16×16) |
| `BETA` | 0.01 | KL divergence weight in β-VAE loss |

### Pre-trained model

`model/scope_model.pth` — trained on OGM-Turtlebot2 (epoch 40, 730K params). Loaded as:
```python
ckpt = torch.load('model/scope_model.pth', map_location=device)
model.load_state_dict(ckpt['model'])  # ckpt also has ['optimizer'] and ['epoch']
```

## Supplementary files

- `CODE_WALKTHROUGH.md` — 492-line detailed code walkthrough in Chinese, covers all 6 source files with diagrams and explanations
- `test_env.py` — quick env + model loading verification
- `quick_demo.py` — fast demo: 3 samples, autoregressive 10-step prediction with 32 Monte Carlo samples, outputs comparison images to `output/quick_mask*.png` and `output/quick_pred*.png`

## Dataset

OGM-Datasets from [Zenodo](https://doi.org/10.5281/zenodo.7051560) (~374 MB compressed). Three subsets: OGM-Turtlebot2 (simulated), OGM-Jackal (real outdoor), OGM-Spot (real indoor). Each has `{train,val,test}/` with `scans/`, `positions/`, `velocities/` subdirectories containing `.npy` files + index `.txt` files.

Pre-trained model is for Turtlebot2 only.

## Environment

`environment.yaml` at project root. Original paper used Python 3.7 + PyTorch 1.7.1; this env uses Python 3.9 + PyTorch 2.7.1 (CUDA 12.8) for compatibility with modern GPUs. Code uses only standard PyTorch APIs — no version migration issues.
