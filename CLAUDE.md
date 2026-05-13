# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

SCOPE (Stochastic Cartographic Occupancy Prediction Engine) — deep learning for occupancy grid map prediction in dynamic environments, IEEE T-RO 2025 (Vol. 41, pp. 4139–4158). arXiv: [2407.00144](https://arxiv.org/abs/2407.00144).

Three model variants:
- **SCOPE++** — full pipeline: robot motion compensation Λ(·) + dynamic object prediction κ(·) (ConvLSTM) + static object segmentation g(·) (GPU Bayesian mapping) + VAE predictor. Highest accuracy.
- **SCOPE** (current branch) — omits static object module g(·); uses only ConvLSTM + VAE. Faster, comparable accuracy.
- **SO-SCOPE** — knowledge-distilled: replaces VAE with a single convolutional layer + uncertainty lookup table. 89× faster than SOTA, 35 FPS on Jetson TX2.

Architecture: **ConvLSTM + β-VAE**. A ConvLSTM encodes 10 frames of local occupancy grids → VAE encoder maps to Gaussian latent space → decoder reconstructs the predicted future occupancy map. Trained with `loss = BCE(pred, gt) + 0.01 × KL(q||p)`.

Paper's formal decomposition (Eq. 3a–3d): the prediction model p_θ(o_{t+1} | d_{t-τ:t}) factorizes into:
- **Λ(d)** — ego-motion compensation: constant velocity model predicts future robot pose, transforms all data to predicted frame R
- **κ(o^R)** — ConvLSTM processes 10-frame OGM sequence for dynamic object prediction
- **g(y^R)** — GPU-accelerated inverse sensor model (Bayesian log-odds update) for static environment map m
- **VAE(z | ô, m)** — reparameterized sampling from learned latent distribution; 32 MC samples averaged for final prediction

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
- **Why Monte Carlo sampling?** VAE is generative — each forward pass samples z ~ N(μ,σ²), producing a different prediction. Averaging 32 samples reduces variance and captures uncertainty.
- **Why knowledge distillation (SCOPE → SO-SCOPE)?** The VAE decoder is memory-intensive (50% of memory, 17% of runtime). The "student" SO-SCOPE replaces it with one conv layer while preserving prediction accuracy via "soft" label training + uncertainty lookup table.

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

## Paper benchmarks (Jetson TX2, 8 GB)

| Metric | ConvLSTM | DeepTracking | PhyDNet | LOPR | SCOPE | SCOPE++ | SO-SCOPE |
|--------|----------|--------------|---------|------|-------|---------|----------|
| FPS | 2.95 | 5.32 | 4.66 | 1.16 | **23.29** | 10.68 | **34.75** |
| Model size (MB) | 12.44 | 0.95 | 37.17 | 1610 | 8.84 | 8.85 | **1.80** |
| Memory (GB) | 0.70 | 0.63 | 0.71 | 5.00 | 0.66 | 0.66 | 0.66 |

Three evaluation metrics (first to use OSPA for OGM prediction):
- **WMSE** (↓): per-cell absolute error, weighted to balance occupied/free cells
- **SSIM** (↑): structural similarity — captures scene geometry preservation
- **OSPA** (↓): optimal subpattern assignment — multi-target tracking metric measuring object count + location errors

## Supplementary files

- `CODE_WALKTHROUGH.md` — detailed code walkthrough in Chinese, covers all 6 source files + paper formula mapping + software optimization explanation
- `test_env.py` — quick env + model loading verification
- `quick_demo.py` — fast demo: 3 samples, autoregressive 10-step prediction with 32 Monte Carlo samples, outputs comparison images to `output/quick_mask*.png` and `output/quick_pred*.png`

## Dataset

OGM-Datasets from [Zenodo](https://doi.org/10.5281/zenodo.7051560) (~374 MB compressed). Three subsets: OGM-Turtlebot2 (simulated), OGM-Jackal (real outdoor), OGM-Spot (real indoor). Each has `{train,val,test}/` with `scans/`, `positions/`, `velocities/` subdirectories containing `.npy` files + index `.txt` files.

Pre-trained model is for Turtlebot2 only.

## Environment

`environment.yaml` at project root. Original paper used Python 3.7 + PyTorch 1.7.1; this env uses Python 3.9 + PyTorch 2.7.1 (CUDA 12.8) for compatibility with modern GPUs. Code uses only standard PyTorch APIs — no version migration issues.
