# MF-LPR2 Research Framework

Multi-frame license plate restoration and recognition (MF-LPR2-inspired), optimized for low-VRAM CUDA (GTX 1650).

## Environment (required)

All commands use the **gemma4** virtual environment at `D:\gemma4\gemma4` (CUDA PyTorch + Numba).

**PowerShell:**
```powershell
. .\scripts\activate_env.ps1
```

**Run a script without activating:**
```powershell
.\scripts\run.ps1 scripts\verify_phase1.py --profile train
```

**Direct Python:**
```powershell
D:\gemma4\gemma4\Scripts\python.exe scripts\verify_phase1.py
```

## Environment path

| Item | Path |
|------|------|
| Venv | `D:\gemma4\gemma4` |
| Activate (PS1) | `D:\gemma4\gemma4\Scripts\Activate.ps1` |
| Python | `D:\gemma4\gemma4\Scripts\python.exe` |

See `configs/environment.yaml` and `.cursor/rules/gemma4-environment.mdc`.

## Phase 1 — verify infrastructure

```powershell
.\scripts\run.ps1 scripts\verify_phase1.py --profile train
```

**MLflow UI:**
```powershell
D:\gemma4\gemma4\Scripts\mlflow.exe ui --backend-store-uri file:///D:/ALPR_Research/outputs/mlruns
```

Metrics are logged with prefixes: `train/`, `val/`, `benchmark/`.

## Phase 2 — RLPR dataset

```powershell
.\scripts\run.ps1 scripts\verify_phase2.py
```

Validates 200 samples, loads multi-frame batches, saves visualizations to MLflow.

## Phase 3 — Optical flow (RAFT-small)

```powershell
.\scripts\run.ps1 scripts\verify_phase3.py
.\scripts\run.ps1 scripts\verify_phase3.py --sample-index 0 --max-pairs 8
```

Outputs are saved under `outputs/experiments/{exp_id}/phases/optical_flow/`:

| Folder | Contents |
|--------|----------|
| `flows/` | `.npy` flow fields |
| `warped/` | Warped frames (PNG) |
| `overlays/` | Reference/warped blends |
| `error_maps/` | Warp error maps |
| `visualizations/` | HSV, magnitude, consistency plots |
| `metrics/` | JSON metrics + report |

## Phase 4 — Temporal fusion

```powershell
.\scripts\run.ps1 scripts\verify_phase4.py --max-pairs 8
```

Compares **mean**, **flow-weighted**, and **attention** fusion under:

`outputs/experiments/{exp_id}/phases/temporal_fusion/`

| Folder | Contents |
|--------|----------|
| `fused/` | Per-method fused PNGs |
| `weights/` | Frame weight JSON |
| `visualizations/` | Comparison panel, attention maps |
| `metrics/` | Method comparison JSON |

## Phase 5 — SwinIR-UNet restoration

```powershell
.\scripts\run.ps1 scripts\verify_phase5.py --max-pairs 8 --fusion-method weighted
```

Outputs under `phases/restoration/`:

| Folder | Contents |
|--------|----------|
| `restored/` | Restored plate PNG |
| `comparisons/` | GT vs center vs restored |
| `error_maps/` | Per-pixel error heatmaps |
| `visualizations/` | PSNR/SSIM charts |
| `metrics/` | JSON quality report |

## Phase 5.1 — Train custom restoration model

Research plan: [docs/phase5_1_training_research_plan.md](docs/phase5_1_training_research_plan.md)

```powershell
. .\scripts\activate_env.ps1
python train_restoration.py
python train_restoration.py --max-epochs 30 --input-mode center_upscale
```

After training, evaluate with your checkpoint:

```powershell
python scripts\verify_phase5.py --checkpoint outputs\experiments\<exp>\checkpoints\epochXX.ckpt
```

MLflow logs `train/loss`, `train/l1`, `train/ssim`, `train/lpips`, `val/psnr`, `val/ssim` each epoch.

## Project layout

See `configs/` for YAML-driven training, inference, dataset, and logging settings.
