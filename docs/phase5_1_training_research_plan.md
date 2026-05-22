# Phase 5.1 — Custom Restoration Training: Research & Integration Plan

## Executive summary

Training `SwinIRUNetHybrid` with **L1 + SSIM + LPIPS** against `Pseudo_GT_ROI` is **necessary and expected to significantly improve** Phase 5 metrics (PSNR/SSIM) and downstream OCR (Phase 6). Your current Phase 5 verify run shows **negative PSNR gain** because weights are **random** — not because the architecture is wrong.

| Current (untrained) | After Phase 5.1 (trained) |
|---------------------|---------------------------|
| PSNR restored ~9 dB | Target **18–25+ dB** on val (dataset-dependent) |
| Worse than center upscale | Should **beat** center baseline consistently |
| OCR unusable on restored | Phase 6 CER/WER should improve |

**Caveat:** RLPR has only **200 samples**. Training only on RLPR risks overfitting. Phase 5.1 is still valuable as:
- Proof of training pipeline (MLflow, checkpoints, reproducibility)
- **Fine-tuning** stage after synthetic / external pretraining
- K-fold or hold-out benchmark for paper-style evaluation

---

## Will it significantly improve performance?

### Yes — for restoration metrics

1. **Supervised signal** — `Pseudo_GT_ROI` is the dataset’s comparison target; L1+SSIM+LPIPS directly optimize what you measure.
2. **Multi-term loss** — L1 preserves pixel accuracy; SSIM preserves structure; LPIPS preserves perceptual similarity (stroke shape).
3. **Global residual learning** — network learns corrections on upsampled fused/center input, stable for small images.

### Moderate — for OCR (Phase 6)

OCR improves **indirectly** via sharper strokes. Large gains need:
- Phase 5.1 restoration training
- Phase 6 PARSeq (optionally fine-tuned on labels)
- Phase 7 OCR-aware loss tying gradients to character regions

### Limited — if training data = RLPR only

| Factor | Impact |
|--------|--------|
| 200 samples | High overfitting risk; use strong aug + early stopping |
| Variable H×W | `batch_size=1`, gradient accumulation |
| Pseudo-GT noise | Ceiling on PSNR; don’t expect perfect GT |
| 31-frame pipeline | Train input: fused latent (Phase 4) preferred over center-only |

**Recommendation:** Treat RLPR as **fine-tune + eval**. Add synthetic plates or external ALPR crops later for pretraining.

---

## Phase 5.1 architecture (training pipeline)

```text
RLPR sample
  → [optional] Phase 3 RAFT + Phase 4 fusion  →  fused latent (CHW)
  → upscale to Pseudo_GT_ROI size
  → SwinIRUNetHybrid
  → restored image
  → Loss vs Pseudo_GT_ROI (L1 + SSIM + LPIPS)
```

**Input modes** (`configs/train_restoration.yaml`):

| Mode | Speed | Fidelity |
|------|-------|----------|
| `center_upscale` | Fast | Baseline training (center frame → GT size) |
| `fused_live` | Slow | Full MF-LPR2 path each step |
| `fused_cache` | Fast after cache | Best for repeated epochs |

---

## Loss design (Phase 5.1 / Phase 7 shared)

```
L_total = w_l1 * L1 + w_ssim * (1 - SSIM) + w_lpips * LPIPS
```

Default weights (from `train.yaml`): `1.0, 0.5, 0.1` — tune on val PSNR.

Phase 7 adds: edge loss, temporal consistency, OCR-aware term (when PARSeq is wired).

---

## Train / validation protocol (RLPR)

- **Split:** 160 train / 40 val (seed=42), or 5-fold CV for papers
- **Metrics logged each epoch:**
  - `train/loss`, `train/l1`, `train/ssim`, `train/lpips`
  - `val/loss`, `val/psnr`, `val/ssim`, `val/psnr_gain_vs_center`
- **MLflow:** all scalars + checkpoint artifacts
- **Checkpoint:** monitor `val/psnr` (max), save top-3 + last
- **Hardware:** gemma4 venv, AMP fp16, `batch_size=1`, `accumulate_grad_batches=4`

---

## Integration with Phase 6 (OCR)

```text
Custom checkpoint (Phase 5.1)
  → configs/restoration.yaml: checkpoint_path: outputs/.../best.ckpt
  → verify_phase5.py / infer.py loads weights
  → Phase 6 PARSeq reads restored PNG
  → Metrics: CER, WER, sequence accuracy vs plate_text_compact
```

**Workflow:** Train restoration → set checkpoint path → run Phase 6 on **restored** images (not center frame).

Optional Phase 6.1: fine-tune PARSeq on RLPR labels with restored images as input.

---

## Integration with Phase 7 (Losses)

| Loss | Phase 5.1 | Phase 7 full |
|------|-----------|--------------|
| L1 | Yes | Yes |
| SSIM | Yes | Yes |
| LPIPS | Yes | Yes |
| Edge | No | Yes |
| Temporal | No | Yes (multi-frame end-to-end) |
| OCR-aware | No | Yes (PARSeq confidence map) |

Phase 7 **extends** `RestorationLoss` — same class, new weights/terms when OCR hook is enabled.

---

## Integration with Phase 8 (full trainer)

Phase 5.1 implements a **focused** `RestorationLightningModule` + `train_restoration.py`.

Phase 8 generalizes to:
- End-to-end fine-tuning (flow + fusion + restoration)
- Ablation flags (CLAHE, attention fusion, OCR loss)

---

## Expected outcomes (realistic)

After 50–100 epochs on RLPR train split (with early stopping):

| Metric | Untrained | Target (val) |
|--------|-----------|--------------|
| PSNR restored | ~9 dB | **20–24 dB** |
| SSIM | ~0.22 | **0.55–0.75** |
| PSNR gain vs center | negative | **+2 to +6 dB** |
| Phase 6 CER | high | **lower** (depends on PARSeq) |

Gains depend on split, augmentation, and input mode (`fused_live` > `center_upscale`).

---

## Implementation checklist (Phase 5.1)

- [x] Research plan (this document)
- [ ] `models/losses/restoration_losses.py`
- [ ] `datasets/rlpr_restoration_dataset.py` (train/val split)
- [ ] `engine/restoration_lightning.py` (LightningModule)
- [ ] `configs/train_restoration.yaml`
- [ ] `train_restoration.py` (CLI, MLflow, gemma4)
- [ ] Update `verify_phase5.py` to load `best.ckpt`
- [ ] `scripts/run_train.ps1` helper

---

## Commands (after implementation)

```powershell
. D:\ALPR_Research\scripts\activate_env.ps1
python train_restoration.py --config configs/train_restoration.yaml
python scripts/verify_phase5.py --checkpoint outputs/experiments/.../checkpoints/best.ckpt
```

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Overfitting 200 samples | Early stopping, conservative aug, k-fold |
| OOM on 4GB GPU | batch_size=1, AMP, gradient checkpointing |
| Slow `fused_live` | Use `center_upscale` first; cache fused tensors |
| LPIPS VRAM | Alex net, single image batches |
| Hallucinated characters | Keep L1 dominant; add OCR loss in Phase 7 |

---

## Conclusion

**Proceed with Phase 5.1 training** — it is the critical step to fix unsatisfying Phase 5 metrics. Phases 6–7 should consume the **trained checkpoint** via config, not random weights. MLflow epoch logging is mandatory for comparing runs and ablations.
