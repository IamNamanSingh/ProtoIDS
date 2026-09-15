# ProtoIDS Progress Summary

**Date:** 2026-09-15 — Branch `main` | Device `RTX 3050 Laptop GPU`

## What Was Achieved

1. **Fixed Ablation (2% → 99%)** — `src/baselines/ablation.py:103` and `src/baselines/ablation_quick.py:175`
   - Root cause: `experiments/data/ciciot_dev.parquet` is already `StandardScaler` scaled (`src/preprocessing/ciciot_preprocessing.py:128`), but scripts re-fit scaler on it (`mean~0, std~1e-06` → rare `1` → `915k` spike) then transformed raw validation (`mean 80M`). Fixed by reusing `results/baselines/ciciot_preprocessor.joblib` + clipping `±5`.
   - Validated: Exp A (41 feats) `0.9912 acc / 0.8240 macroF1`, Exp B (34 feats) `0.9910 / 0.8069` (`results/baselines/ablation_results.json:3`).

2. **Fixed ProtoIDS Training (1% → 84%)** — `src/protoids/dataset.py:103`, `src/protoids/train_protoids.py:412`, `src/protoids/training.py:64`
   - Double scaling on train `max 915k` vs val `15k` → `Linear 41→128` (`src/protoids/encoder.py:33`) saturation, `BatchNorm` divergence.
   - `model.to(device)` before KMeans, `normalized_embedding` for init, `similarities = -16*distance` for sharp CrossEntropy, clip `±5`, fixed `.cpu()` for open-set `src/protoids/train_protoids.py:200`.
   - Now `train 0.85→0.58`, `val 0.48→0.39` (was `4→8`).

3. **Closed-set vs RF Baseline** — `src/protoids/train_protoids.py:491`
   - Best ProtoIDS (K=3, 25 epochs, BS 1024, `λ=0.01`): **Val 0.8402 acc / 0.5976 macroF1 / 0.8263 MCC**, Test `0.8391/0.5956` (`experiments/results/protoids_k3_clip/`).
   - RF baseline: `0.9912 / 0.8240 / 0.9904` (`reports/dataset_forensic_report.md:18`, `results/baselines/baseline_experiment_results.json`). Gap `~15pp` remains.

4. **K Sweep (1,3,5)** — `reports/protoids_methodology.md:28`
   - K=1: `0.8448 / 0.5828 / AUROC 0.9082` (best accuracy)
   - K=3: `0.8402 / 0.5976 / AUROC 0.9366` (best balance, recommended)
   - K=5: `0.8342 / 0.5757 / AUROC 0.9307`

5. **Open-set (withhold MITM/VulnScan/BruteForce)** — `src/protoids/train_protoids.py:604` `τ=90th pct` (`HANDOVER.md:29`)
   - K=3: `AUROC 0.9366`, `Known 0.856`, `Unknown Recall 0.87`, but `FAR 0.937 / Prec 0.06` → threshold too permissive. Current eval is leaky (trained on 34, eval split only); true open-set needs retrain on 31 classes.

## Final Accuracy

| Model | Closed Val Acc | Macro F1 | Weighted F1 | AUROC (open) |
|-------|---------------|----------|-------------|--------------|
| **RF (baseline, 100 trees)** | **0.9912** | **0.8240** | 0.9916 | — |
| ProtoIDS K=1 (clip) | 0.8448 | 0.5828 | 0.8391 | 0.908 |
| **ProtoIDS K=3 (clip) — recommended** | **0.8402** | **0.5976** | 0.8370 | **0.937** |
| ProtoIDS K=5 (clip) | 0.8342 | 0.5757 | 0.8271 | 0.931 |
| ProtoIDS initial (bug, clip_fix) | 0.8285 | 0.5681 | 0.8196 | 0.935 |

**Artifacts:** `experiments/models/protoids_k3_clip/model.pth`, `results/baselines/ablation_results.json`, `experiments/data/ciciot_dev.parquet:235k`.

## Remaining

Cross-dataset (`Edge-IIoTset`), true open-set retrain (31 classes), SOTA comparison, threshold sweep (`p=95/99`), frontend demo.
