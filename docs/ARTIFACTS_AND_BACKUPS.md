# Artifacts and Backups Inventory

This document inventories the experimental artifacts and backups stored in the `backups/` directory.

## CICIoT2023 Experiments

### Reference K3 Experiment
- **Archive**: `backups/CICIoT2023/reference_K3/ProtoIDS_FINAL_K3.zip`
- **Experiment Identity**: Final CICIoT2023 true-open-set experiment with K=3, lambda=0.1
- **Model Files**:
  - `protoids_final/model.pth`
  - `protoids_final/best_model.pth`
- **Best Checkpoint**: `protoids_final/best_model.pth` (appears to be the best model based on validation)
- **Prototype Vectors**: `protoids_final/prototypes.npy`
- **Threshold**: `protoids_final/threshold.npy`
- **Label Mapping**: 
  - `protoids_final/results/ciciot_label_mapping.json`
  - `protoids_final/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: `protoids_final/results/ciciot_preprocessor.joblib`
- **Results**:
  - `protoids_final/results/training_history.json`
  - `protoids_final/results/closed_set_results.json`
  - `protoids_final/results/open_set_results.json`
  - `protoids_final/results/open_set_test_results.json`
- **Analysis Files**: `protoids_final/analyze_threshold.py`
- **Logs**: None explicitly stored (but training history may contain logs)

### Ablation K=1 Experiment
- **Archive**: `backups/CICIoT2023/ablation_K1/ProtoIDS_K1_MASTER_BASELINE.zip`
- **Experiment Identity**: CICIoT2023 ablation with K=1, lambda=0.1
- **Model Files**:
  - `protoids_k1/model.pth`
- **Best Checkpoint**: Not explicitly separated; `model.pth` may be the final model
- **Prototype Vectors**: `protoids_k1/prototypes.npy`
- **Threshold**: `protoids_k1/threshold.npy`
- **Label Mapping**: 
  - `protoids_k1/results/ciciot_label_mapping.json`
  - `protoids_k1/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: `protoids_k1/results/ciciot_preprocessor.joblib`
- **Results**:
  - `protoids_k1/results/training_history.json`
  - `protoids_k1/results/closed_set_results.json`
  - `protoids_k1/results/open_set_results.json`
  - `protoids_k1/results/open_set_test_results.json`
- **Analysis Files**: None
- **Logs**: None

### Ablation K=3 No Compactness Experiment
- **Archive**: `backups/CICIoT2023/ablation_K3_no_compact/ProtoIDS_K3_NOCOMPACT.zip`
- **Experiment Identity**: CICIoT2023 ablation with K=3, lambda=0 (no compactness loss)
- **Model Files**:
  - `protoids_k3_nocompact/model.pth`
- **Best Checkpoint**: Not explicitly separated
- **Prototype Vectors**: `protoids_k3_nocompact/prototypes.npy`
- **Threshold**: `protoids_k3_nocompact/threshold.npy`
- **Label Mapping**: 
  - `protoids_k3_nocompact/results/ciciot_label_mapping.json`
  - `protoids_k3_nocompact/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: `protoids_k3_nocompact/results/ciciot_preprocessor.joblib`
- **Results**:
  - `protoids_k3_nocompact/results/training_history.json`
  - `protoids_k3_nocompact/results/closed_set_results.json`
  - `protoids_k3_nocompact/results/open_set_results.json`
  - `protoids_k3_nocompact/results/open_set_test_results.json`
- **Analysis Files**: `protoids_k3_nocompact/analyze_threshold.py`
- **Logs**: None

### Ablation K=5 Experiment
- **Archive**: `backups/CICIoT2023/ablation_K5/ProtoIDS_K5_MASTER_BASELINE.zip`
- **Experiment Identity**: CICIoT2023 ablation with K=5, lambda=0.1
- **Model Files**:
  - `protoids_k5/model.pth`
- **Best Checkpoint**: Not explicitly separated
- **Prototype Vectors**: `protoids_k5/prototypes.npy`
- **Threshold**: `protoids_k5/threshold.npy`
- **Label Mapping**: 
  - `protoids_k5/results/ciciot_label_mapping.json`
  - `protoids_k5/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: `protoids_k5/results/ciciot_preprocessor.joblib`
- **Results**:
  - `protoids_k5/results/training_history.json`
  - `protoids_k5/results/closed_set_results.json`
  - `protoids_k5/results/open_set_results.json`
  - `protoids_k5/results/open_set_test_results.json`
- **Analysis Files**: `protoids_k5/analyze_threshold.py`
- **Logs**: None

## Edge-IIoTset Experiments

### Final Experiment
- **Archive**: `backups/Edge-IIoTset/final_experiment/ProtoIDS_EdgeIIoT_Final_Experiment.zip`
- **Experiment Identity**: Edge-IIoTset final open-set experiment (edgeiiot_open_set_k3_lambda01_run3)
- **Model Files**:
  - `model/model.pth`
  - `model/best_model.pth`
- **Best Checkpoint**: `model/best_model.pth`
- **Prototype Vectors**: `model/prototypes.npy`
- **Threshold**: `model/threshold.npy`
- **Label Mapping**: `model/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: None explicitly stored (but note: the scaler might be in the analysis or not saved)
- **Results**: None explicitly stored in the ZIP (but see analysis file below)
- **Analysis Files**: `analysis/ProtoIDS_EdgeIIoT_Threshold_Analysis_Final.txt` (detailed threshold analysis)
- **Logs**: None

### Model Backup
- **Archive**: `backups/Edge-IIoTset/model_backup/ProtoIDS_EdgeIIoT_Model_Backup.zip`
- **Experiment Identity**: Appears to be the same experiment as the final experiment (edgeiiot_open_set_k3_lambda01_run3) but only model artifacts
- **Model Files**:
  - `edgeiiot_open_set_k3_lambda01_run3/model.pth`
  - `edgeiiot_open_set_k3_lambda01_run3/best_model.pth`
- **Best Checkpoint**: `edgeiiot_open_set_k3_lambda01_run3/best_model.pth`
- **Prototype Vectors**: `edgeiiot_open_set_k3_lambda01_run3/prototypes.npy`
- **Threshold**: `edgeiiot_open_set_k3_lambda01_run3/threshold.npy`
- **Label Mapping**: `edgeiiot_open_set_k3_lambda01_run3/label_mapping.json`
- **Preprocessing/Scaler Artifacts**: None
- **Results**: None
- **Analysis Files**: None
- **Logs**: None

## Unclassified Artifacts
- **Archive**: `backups/unclassified/` (currently empty)
- **Note**: No unclassified artifacts found at this time.

## Additional Files
- **Threshold Analysis Text File**: 
  - Original: `backups/ProtoIDS_EdgeIIoT_Threshold_Analysis_Final.txt`
  - Copy: `docs/experiments/Edge-IIoTset/ProtoIDS_EdgeIIoT_Threshold_Analysis_Final.txt`
  - **Content**: Detailed threshold analysis for the Edge-IIoTset experiment.

## Notes
- All model files are in PyTorch format (`.pth`).
- Prototype vectors and thresholds are stored as NumPy arrays (`.npy`).
- Label mappings are JSON files.
- Preprocessing scalers are stored as joblib files.
- Results include training history and evaluation metrics (closed-set and open-set).
- The CICIoT2023 experiments include an `analyze_threshold.py` script for threshold sweeping.

## Verification
- The ZIP archives were inspected using `unzip -l` and the contents recorded above.
- No extraction was performed to avoid altering the original backups.

