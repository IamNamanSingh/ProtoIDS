# Experiment Log

## Chronological Experiment Log for ProtoIDS

This document provides a high-level chronological log of major experiments and milestones in the ProtoIDS project.

### Early Exploration and Baseline Establishment
- **Initial Dataset Acquisition**: Obtained CICIoT2023 and Edge-IIoTset datasets
- **Forensic Analysis Completed**: 
  - CICIoT2023: Feature inspection, label analysis, class imbalance, duplicate analysis, leakage analysis
  - Edge-IIoTset: Forensic analysis completed with leakage concerns documented
  - Cross-dataset feature-space differences analyzed
- **Preprocessing Pipeline Established**: 
  - Identification of 41 retained numerical features after removing leakage-prone columns (IPs, ports, timestamps, etc.)
  - Development subset created at `experiments/data/ciciot_dev.parquet` (~235k rows) for rapid experimentation
- **Baseline Models Evaluated**: 
  - Random Forest, Gradient Boosting, MLP evaluated on CICIoT2023 development subset
  - Results: Random Forest (Acc: 0.8400, Macro F1: 0.6684), Gradient Boosting (Acc: 0.8220, Macro F1: 0.6385), MLP (Acc: 0.7985, Macro F1: 0.6012)

### ProtoIDS Architecture Implementation
- **Core Architecture Implemented**:
  - Encoder: Linear-BatchNorm-ReLU-Dropout-Linear-Relex-Linear-L2 normalization
  - Prototype Layer: Learnable prototypes per class with cosine distance
  - Unknown detection via distance threshold
- **Initial Experiments**:
  - Initial closed-set experiments conducted
  - Initial open-set implementation created (with known issues)

### Critical Protocol Corrections
- **Label Mapping Fix**: Fixed NoneType error in label mapping logic
- **Scaler Fitting Protocol Fix**: 
  - Corrected to fit StandardScaler ONLY on known training data (after withholding unknown classes)
  - Fixed AttributeError in preprocessing pipeline
- **Unit Tests and Integration Testing**: 
  - Unit tests for label mapping logic created and passed
  - Integration testing of corrected pipeline performed

### Open-Set Experiments and Ablation Studies (Development Subset)
- **Open-set experiments conducted on development subset** with corrected protocol
  - Experiments used `--withhold_open_set` flag to withhold:
    - CICIoT2023: MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce
    - Edge-IIoTset: MITM, Password, Vulnerability_scanner
  - Threshold calculated as 90th percentile of minimum prototype distance on known validation samples
- **Ablation Studies Performed** (development subset):
  - Varying K (number of prototypes per class): K=1, K=3, K=5
  - Varying compactness lambda: lambda=0.1 (reference), lambda=0.0 (no compactness)
  - Results documented for each configuration

### Memory-Efficient Processing Preparation
- **Memory Limitation Identified**: 
  - Full CICIoT2023 processing (5.5M rows) fails during feature scaling due to memory allocation error
  - Error: `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.66 GiB for an array with shape (5449718, 41) and data type float64`
- **Root Cause**: Intermediate float64 array during clipping operation (`np.clip(X_train_scaled, -5, 5).astype(np.float32)`)
- **Classification**: Computational/memory limitation, not scientific or model failure
- **Required Solution**: Implement memory-efficient preprocessing while preserving exact scientific protocol:
  - Withholding must occur BEFORE any scaler fitting
  - Scaler fitted ONLY on known training data (after withholding)
  - Validation/test transformation must use scaler fitted exclusively on known training data

### Documentation and Artifact Organization
- **Backups Organized**: 
  - Created structured backup directory:
    - `backups/CICIoT2023/reference_K3/`
    - `backups/CICIoT2023/ablation_K1/`
    - `backups/CICIoT2023/ablation_K3_no_compact/`
    - `backups/CICIoT2023/ablation_K5/`
    - `backups/Edge-IIoTset/final_experiment/`
    - `backups/Edge-IIoTset/model_backup/`
    - `backups/unclassified/`
  - Moved and copied backup files accordingly
  - Kept original backup copies where useful
- **Documentation Created**:
  - `docs/ARTIFACTS_AND_BACKUPS.md`: Detailed inventory of experimental backups
  - `docs/EXPERIMENT_LOG.md`: This chronological experiment log
  - `docs/REPRODUCIBILITY.md`: Reproducibility guidelines and verification steps
  - `docs/RESEARCH_DECISIONS.md`: Key research decisions and justifications
  - `docs/FRONTEND_HANDOVER.md`: Guidelines for frontend/demo development
  - `docs/NEXT_PERSON_XIIOTID.md`: Specific instructions for X-IIoTID dataset handling
- **README Updated**: 
  - Comprehensive update with project status, architecture, results, limitations, and handoff instructions
  - Added sections for final documented results, ablations, tradeoffs, and known unresolved tasks
- **Result Files Preserved**: 
  - Extracted and preserved CICIoT2023 reference K3 result files to `results/documented/CICIoT2023/`
  - Copied Edge-IIoTset threshold analysis to `docs/experiments/Edge-IIoTset/` and considered for `results/documented/Edge-IIoTset/`

### Immediate Next Steps
1. **Implement memory-efficient preprocessing** in `src/protoids/dataset.py` (chunked processing or incremental StandardScaler fitting)
2. **Run clean K=3 open-set experiment** with memory-efficient implementation
3. **Verify leakage audit** passes with new implementation
4. **Proceed with K=1/3/5 experiments** and threshold sensitivity analysis
5. **Begin X-IIoTID dataset work** following `docs/NEXT_PERSON_XIIOTID.md`

## Notes
- This log is updated periodically; see git commit history for detailed code changes.
- All experiments use random seed 42 for reproducibility unless otherwise noted.
- The true open-set protocol must be preserved in all future work.

---
*Last updated: $(date +%Y-%m-%d)*