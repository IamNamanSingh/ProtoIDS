# ProtoIDS: Prototype-Based Open-Set IoT Intrusion Detection

**Prototype-Based Open-Set IoT Intrusion Detection**

## What is ProtoIDS?
ProtoIDS is an experimental deep learning approach for IoT intrusion detection. Traditional classifiers simply draw boundaries between known attacks, which causes them to confidently misclassify unseen/zero-day attacks as "normal" or some known attack type. 

ProtoIDS solves this by learning "prototypes" (representative embeddings) for network behaviors. When an attack comes in, the model measures its cosine distance to these prototypes. If the attack is too far from any known prototype, it is flagged as **UNKNOWN**, rather than being misclassified.

## Project Status
**Current Phase: Final Research Handoff**

This repository represents the finalized research state of ProtoIDS after completing:
- True open-set experiments on CICIoT2023 (reference K=3, lambda=0.1)
- Ablation studies on CICIoT2023 (K=1, K=3 no compactness, K=5)
- True open-set experiment on Edge-IIoTset (K=3, lambda=0.1)
- Threshold analysis for both datasets
- Memory-efficient preprocessing implementations
- Comprehensive documentation and artifact organization

What remains for the next researcher:
- Third dataset (X-IIoTID) acquisition, preprocessing, and true-open-set evaluation
- Cross-dataset generalization studies (CICIoT2023 ↔ Edge-IIoTset ↔ X-IIoTID)
- Frontend/demo for visualization and interaction
- Final research paper/report compilation

## Core Research Idea
Network flow features → Encoder → L2-normalized embedding → Multiple learnable prototypes per known class → Cosine distance → Nearest known class → Distance threshold → KNOWN / UNKNOWN decision

## True Open-Set Protocol (MUST FOLLOW)
1. **Withhold Unknown Classes**: During training, specific attack classes are completely withheld from the training data
2. **Fit Scaler on Known Data Only**: StandardScaler is fitted only on known class training samples (after withholding)
3. **Learn Prototypes for Known Classes Only**: Prototype vectors are learned exclusively from known class data
4. **Threshold Calibration**: Distance threshold for UNKNOWN detection is calibrated using validation set known-class distances (using the project's documented convention: True FAR = fraction of true unknown samples incorrectly accepted as known)
5. **Unknown Detection**: Samples with maximum prototype distance exceeding threshold are flagged as UNKNOWN
6. **Validation Separation**: Threshold is calibrated on validation set, then reused for test set evaluation (never recalibrated on test)

## Key Features
- **Leakage-Free Preprocessing**: Comprehensive forensic analysis ensures identifiers like IP addresses, timestamps, and raw payloads are removed
- **Memory-Efficient Processing**: Implements chunked data processing and incremental scaling to handle massive datasets
- **Rigorous Open-Set Protocol**: Explicitly withholds specific attack classes during training to evaluate zero-day attack detection
- **Correct Unknown Class Handling**: Prototypes learned only for known classes; UNKNOWN detected via distance-based rejection
- **Development Subset Capability**: Option to use curated subset for rapid prototyping
- **Cross-Dataset Validation**: Designed to evaluate generalization across different IoT environments
- **Checkpointing and Metrics**: Training loop saves best model checkpoints and records comprehensive metrics

## Project Structure
- `src/preprocessing/`: Memory-efficient chunked data processing, feature scaling, and open-set protocol enforcement
  - `ciciot_preprocessing.py`: For CICIoT2023 dataset
  - `edgeiiot_preprocessing.py`: For Edge-IIoTset dataset
- `src/baselines/`: Scripts for traditional ML baselines (Random Forest, Gradient Boosting, MLP) and ablation studies
- `src/protoids/`: Core ProtoIDS architecture
  - `encoder.py`: Feature encoder network
  - `prototype_layer.py`: Learnable prototypes with cosine distance computation
  - `protoids_model.py`: Main ProtoIDS model
  - `dataset.py`: Memory-efficient dataset loading with development subset support
  - `training.py`: Training loop with unknown class masking and checkpointing
  - `train_protoids.py`: Main training script with open-set evaluation
  - `autoencoder.py`: Utility for pretraining encoder
  - `pretrain_ae.py`: Script for autoencoder pretraining
  - `cross_evaluate.py`: Utility for cross-dataset evaluation (CICIoT → Edge-IIoT)
- `analysis/`: Threshold analysis tools and scripts
- `reports/`: Analysis reports, methodology documentation, forensic analysis
- `experiments/`: 
  - `configs/`: Experiment configurations (JSON/YAML)
  - `results/`: Experiment logs, results, and metrics
  - `models/`: Saved model checkpoints (only lightweight configs, not large files)
  - `processed/`: Preprocessed data (memmap arrays, development subsets)
- `docs/`: Comprehensive documentation
  - `ARTIFACTS_AND_BACKUPS.md`: Inventory of experimental backups
  - `EXPERIMENT_LOG.md`: Chronological experiment log
  - `REPRODUCIBILITY.md`: Reproducibility guidelines
  - `RESEARCH_DECISIONS.md`: Key research decisions and justifications
  - `FRONTEND_HANDOVER.md`: Guidelines for frontend/demo development
  - `NEXT_PERSON_XIIOTID.md`: Specific instructions for X-IIoTID dataset handling
  - `experiments/CICIoT2023/`: CICIoT2023 experiment documentation and results
  - `experiments/Edge-IIoTset/`: Edge-IIoTset experiment documentation and results
- `results/documented/`: Final documented results for publications
  - `results/documented/CICIoT2023/`
  - `results/documented/Edge-IIoTset/`
- `README.md`: This file
- `HANDOVER.md`: Detailed handover documentation for continuation
- `requirements.txt`: Python dependencies

## Setup Instructions
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Directory Structure
The repository automatically creates necessary directories for experiments:
- `experiments/results/` - for experiment logs and results
- `experiments/models/` - for saved model checkpoints (lightweight configs only)
- `experiments/processed/` - for processed data (memmap arrays, development subsets)
- `experiments/configs/` - for experiment configurations

### 3. Development Subset
To use the development subset for rapid prototyping, ensure `experiments/data/ciciot_dev.parquet` exists (created by preprocessing script).

### 4. Reproducibility
- All experiments use random seed 42 for reproducibility
- Preprocessing scripts are deterministic when given the same input
- Model checkpoints should be evaluated with the same preprocessing pipeline
- See `docs/REPRODUCIBILITY.md` for detailed guidelines

## Experiment Configurations
- CICIoT2023 reference: K=3, lambda=0.1, dropout=0.2, lr=0.001, batch=256, epochs=10, seed=42
- CICIoT2023 ablation K=1: K=1, lambda=0.1, otherwise same as reference
- CICIoT2023 ablation K=3 no compactness: K=3, lambda=0, otherwise same as reference
- CICIoT2023 ablation K=5: K=5, lambda=0.1, otherwise same as reference
- Edge-IIoTset: K=3, lambda=0.1, dropout=0.2, lr=0.001, batch=256, epochs=10, seed=42
- See `docs/experiments/CICIoT2023/` and `docs/experiments/Edge-IIoTset/` for detailed configs

## Final Documented Results
### CICIoT2023 Reference Experiment (K=3, lambda=0.1)
- **Known Classes**: 31 (withheld: MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce)
- **Known Training Samples**: 5,449,718
- **Unknown Validation**: 8,936
- **Unknown Test**: 9,072
- **Input Features**: 41
- **Embedding**: 32-dimensional L2-normalized
- **Prototypes**: 3 per known class
- **Dropout**: 0.2
- **Learning Rate**: 0.001
- **Compactness Lambda**: 0.1
- **Batch Size**: 256
- **Epochs**: 10
- **Seed**: 42

**Test Results:**
- Accuracy: 0.9610
- Macro F1: 0.6427
- Weighted F1: 0.9562
- Macro Recall: 0.6478
- MCC: 0.9577
- Known Accuracy: 0.9897
- Known Macro F1: 0.7628
- Known Recall: 0.7494
- Unknown Precision: 0.0598
- Unknown Recall: 0.8209
- Unknown F1: 0.1114
- AUROC: 0.9468
- AUPR: 0.1246
- Threshold: 0.0364
- True FAR: 0.1791 (1 - Unknown Recall)

### CICIoT2023 Ablation Studies
**K=1, lambda=0.1:**
- Accuracy: 0.9493
- Macro F1: 0.6418
- MCC: 0.9450
- Unknown Recall: 0.7470
- Unknown F1: 0.1021
- AUROC: 0.9259
- AUPR: 0.1172

**K=5, lambda=0.1:**
- Accuracy: 0.9617
- Macro F1: 0.6393
- MCC: 0.9581
- Unknown Recall: 0.7760
- Unknown F1: 0.1053
- AUROC: 0.9236
- AUPR: 0.1539

**K=3, lambda=0 (no compactness):**
- Accuracy: 0.9682
- Macro F1: 0.6465
- MCC: 0.9653
- Unknown Recall: 0.5874
- Unknown F1: 0.0809
- AUROC: 0.8770
- AUPR: 0.0628

### Edge-IIoTset Experiment (edgeiiot_open_set_k3_lambda01_run3)
- **Known Classes**: 12 (withheld: MITM, Password, Vulnerability_scanner)
- **Input Features**: 39
- **Embedding**: 32-dimensional L2-normalized
- **Prototypes**: 3 per known class
- **Dropout**: 0.2
- **Learning Rate**: 0.001
- **Lambda**: 0.1
- **Batch Size**: 256
- **Epochs**: 10
- **Seed**: 42

**Closed-Set Test:**
- Accuracy: 0.9250
- Macro F1: 0.6994
- Weighted F1: 0.9062
- Macro Recall: 0.7017
- MCC: 0.8346
- Saved threshold: 0.0571

**Open-Set Test at 0.0571:**
- Known Accuracy: 0.8920
- Known Macro F1: 0.6016
- Known Recall: 0.5291
- Unknown Precision: 0.2678
- Unknown Recall: 0.7686
- Unknown F1: 0.3972
- True FAR: 0.2314
- KRR: 0.1007
- AUROC: 0.9281
- AUPR: 0.3847

**Validation Max-Unknown-F1 Threshold (0.0970):**
- Unknown F1: 0.4982
- Unknown Precision: 0.4164
- Unknown Recall: 0.6202
- True FAR: 0.3798
- KRR: 0.0417
- Known Acceptance: 0.9583
- Known Macro F1: 0.6942

**Fixed Test at Threshold 0.0970:**
- Known Accuracy: 0.9465
- Known Macro F1: 0.7072
- Known Recall: 0.6496
- Unknown Precision: 0.4167
- Unknown Recall: 0.6221
- Unknown F1: 0.4991
- True FAR: 0.3779
- KRR: 0.0417
- AUROC: 0.9281
- AUPR: 0.3847

## Ablations and Tradeoffs
- Increasing K from 1 to 5 shows slight variations in macro F1 (0.6418 → 0.6393) but improved unknown recall (0.7470 → 0.7760)
- Removing compactness lambda (K=3, lambda=0) improves known-class accuracy but reduces unknown recall (0.5874 vs 0.8209 in reference)
- Lambda controls the tradeoff between compactness and separation; lambda=0.1 provides balanced performance
- No single configuration is universally optimal; choice depends on application priorities (known-class vs unknown-class detection)

## Limitations
- **Computational Constraints**: Full CICIoT2023 processing requires memory-efficient implementation; very large batches may still challenge limited RAM
- **Dataset Scope**: Currently validated on two IoT datasets (CICIoT2023 and Edge-IIoTset); cross-dataset generalization requires further study
- **Threshold Sensitivity**: Unknown class detection performance varies with threshold choice; optimal threshold is dataset-dependent
- **Feature Representation**: Relies on quality of network flow features; may not capture all behavioral nuances in complex attacks
- **Single-Run Experiments**: Results represent single training runs with seed 42; statistical significance requires multiple runs
- **Open-Set Interpretation**: UNKNOWN flag indicates deviation from known prototypes, not semantic understanding of attack type

## Known Unresolved Tasks
1. **Third Dataset (X-IIoTID)**:
   - Obtain X-IIoTID dataset from authoritative source
   - Inspect schema and audit for leakage
   - Analyze class imbalance and define true-open-set classes
   - Justify withheld-class selection based on zero-day relevance
   - Build dataset-specific preprocessing pipeline
   - Fit scaler only on known training data
   - Train ProtoIDS and evaluate with proper open-set protocol
   - Run threshold analysis and preserve artifacts
   - Document limitations and findings

2. **Cross-Dataset Generalization**:
   - Align features/protocols between CICIoT2023, Edge-IIoTset, and X-IIoTID
   - Train on one dataset, test on another (and combinations)
   - Evaluate performance drop and analyze causes
   - Document whether architectural changes are needed for generalization

3. **Frontend/Demo**:
   - Implement lightweight research demo for ProtoIDS inference
   - Show input flow, predicted class, nearest known class, prototype distance, threshold, and KNOWN/UNKNOWN decision
   - Include analytics: confusion matrix, distance distributions, threshold curve, attack-wise metrics, unknown absorption
   - Avoid production dashboard complexity; focus on interpretability and research utility

4. **Final Research Paper/Report**:
   - Compile methods, results, and discussions into academic format
   - Include detailed experimental protocols, ablation studies, and limitation analysis
   - Prepare for submission to relevant venues or internal reporting

## Handoff Instructions
The next researcher should:
1. **Review Documentation**: Start with `docs/EXPERIMENT_LOG.md` and `docs/REPRODUCIBILITY.md`
2. **Verify Artifacts**: Check `docs/ARTIFACTS_AND_BACKUPS.md` for backup inventory
3. **Reproduce Baseline**: Run development subset experiments to verify setup
4. **Proceed to X-IIoTID**: Follow `docs/NEXT_PERSON_XIIOTID.md` for third dataset work
5. **Cross-Dataset Validation**: After X-IIoTID, perform cross-dataset studies
6. **Frontend/Demo**: Implement research demo as time permits
7. **Write Report**: Compile findings into final documentation
8. **Commit Regularly**: Use descriptive commit messages and push to `origin/protoids-next`
9. **Do Not Commit Large Files**: Raw datasets, model checkpoints, and large arrays belong in `backups/` (which is ignored)
10. **Preserve Scientific Protocol**: Never compromise the true open-set protocol (withholding → fitting → transforming) for convenience

## References
For detailed forensic analysis, see:
- `reports/dataset_forensic_report.md`
- `reports/protoids_methodology.md`

For baseline results and experimental protocols, see:
- `reports/baseline_experiment_report.md`
- `reports/progress_summary.md`

## Next Steps (Immediate)
1. Validate the repository structure and documentation
2. Run lightweight verification (see `docs/REPRODUCIBILITY.md`)
3. Begin X-IIoTID dataset acquisition and inspection