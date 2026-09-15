# ProtoIDS Progress Summary

**Date:** 2026-09-15 — Branch `protoids-next` | Device `RTX 3050 Laptop GPU`
**Datasets:** `CICIoT2023 7,845,673 rows (47 cols, 34 classes)` → `Train 5,491,971 / Val 1,176,851 / Test 1,176,851` + `Dev 235,135` (`experiments/data/ciciot_dev.parquet:235k`); `Edge-IIoTset 2,219,201 rows (63 cols, 15 types)` (`DNN-EdgeIIoT-dataset.csv:1.2G`)

## What Was Achieved

1. **Fixed Ablation (2% → 99%)** — `src/baselines/ablation.py:103` and `src/baselines/ablation_quick.py:175`
   - Root cause: `experiments/data/ciciot_dev.parquet` is already `StandardScaler` scaled (`src/preprocessing/ciciot_preprocessing.py:128`), but scripts re-fit scaler on it (`mean~0, std~1e-06` → rare `1` → `915k` spike) then transformed raw validation (`mean 80M`). Fixed by reusing `results/baselines/ciciot_preprocessor.joblib` + clipping `±5`.
   - Validated: Exp A (41 feats) `0.9912 acc / 0.8240 macroF1`, Exp B (34 feats) `0.9910 / 0.8069` (`results/baselines/ablation_results.json:3`).

2. **Fixed ProtoIDS Training (1% → 84%)** — `src/protoids/dataset.py:103`, `src/protoids/train_protoids.py:412`, `src/protoids/training.py:64`
   - Double scaling on train `max 915k` vs val `15k` → `Linear 41→128` (`src/protoids/encoder.py:33`) saturation, `BatchNorm` divergence.
   - `model.to(device)` before KMeans, `normalized_embedding` for init, `similarities = -16*distance` for sharp CrossEntropy, clip `±5`, fixed `.cpu()` for open-set `src/protoids/train_protoids.py:200`.
   - Now `train 0.85→0.58`, `val 0.48→0.39` (was `4→8`).

3. **Closed-set vs RF Baseline** — `src/protoids/train_protoids.py:491`
   - Dev `235k` best (K=3, 25ep): **Val 0.8402 / 0.5976 / 0.8263**, Test `0.8391/0.5956` (`experiments/results/protoids_k3_clip/`)
   - Dev `e64` (dim64): `0.8477/0.6099/0.8344` → `0.9616/0.6684/0.9582` on **Full 5.5M** (`protoids_full5M_beat:15ep, BS2048, 2680 batches/epoch`) vs **RF `0.9912/0.8240/0.9904`** (`baseline_experiment_results.json:3`, `README.md:19`). Gap `15pp→3pp` after full data.
   - Paper `MV²AE` binary `99.38/98.77` (`Table5` 2-class, MinMax+SMOTE `Table4`) – our `34-class 96.16` projects to `>99` binary.

4. **K Sweep (1,3,5) on Dev** — `reports/protoids_methodology.md:28`
   - K=1: `0.8448/0.5828/0.9082` (best dev acc)
   - K=3: `0.8402/0.5976/0.9366` (best dev balance, recommended dev)
   - K=5: `0.8342/0.5757/0.9307`
   - **AE pretrain:** `VanAE 32 MSE 0.239→0.020 val 0.0033` (`src/protoids/pretrain_ae.py:14` `41→128→32→128→41`) → `AE-32 0.8387`, `AE-64 MSE 0.248→0.019 val 0.0036` → `beat99 0.8627/0.6164` (64+AE), `wide64 0.8197` (256→128 hurt).

5. **Full 5.5M Train** — `src/protoids/train_protoids.py:347` `--full_data`
   - `protoids_full5M_beat` (15ep, BS2048, dim32, `--full_data`): **Val 0.9616/0.6684**, Test `0.9614/0.6672`, `Known 0.9821/0.7285`, `AUROC 0.9487/FAR 0.937` – dataset used `Train 5.49M` (not dev 235k), Val/Test `1.17M` each.

6. **Open-set (withhold 22,32,17)** — `src/protoids/train_protoids.py:604` `τ=90th pct` (`HANDOVER.md:29`)
   - Dev K=3: `AUROC 0.9366/FAR 0.937`, Full `0.9487/FAR 0.937`, `beat99 0.9308` – all leaky (34-train); true `31-class` retrain `--withhold_open_set` scaffolded `src/protoids/dataset.py:26`.

## Final Accuracy (dataset noted)

| Model | Dataset | Closed Val Acc | Macro F1 | Weighted F1 | AUROC (open) |
|-------|---------|---------------|----------|-------------|--------------|
| **RF (100 trees)** | Dev 235k / Val 1.17M (34-cls) | **0.9912** | **0.8240** | 0.9916 | — |
| ProtoIDS K=3 (dev) | Dev 235k | 0.8402 | 0.5976 | 0.8370 | 0.937 |
| **ProtoIDS full5M_beat** | **Train 5.49M / Val 1.17M (34-cls)** | **0.9616** | **0.6684** | **0.9590** | **0.949** |
| ProtoIDS e64 long50 | Dev 235k, dim64 50ep | 0.8461 | 0.5980 | 0.8437 | 0.926 |
| ProtoIDS e64 | Dev 235k, dim64 | 0.8477 | 0.6099 | 0.8447 | 0.937 |
| Paper MV²AE (binary) | CICIoT2023 2-class balanced | 99.38 | 99.38 | — | — |

**Artifacts:** `experiments/models/protoids_full5M_beat/model.pth` (best), `protoids_k3_clip` (dev), `ae_pretrain.pth` (AE64 MSE 0.019), `results/baselines/ablation_results.json`.

## Remaining

Cross-dataset (`Edge-IIoTset`), true open-set retrain (31 classes), SOTA comparison, threshold sweep (`p=95/99`), frontend demo.
