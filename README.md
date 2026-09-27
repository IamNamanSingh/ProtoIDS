# ProtoIDS: Prototype-Based Open-Set IoT Intrusion Detection

**Prototype-Based Open-Set IoT Intrusion Detection**

## What is ProtoIDS?
ProtoIDS is an experimental deep learning approach for IoT intrusion detection. Traditional classifiers simply draw boundaries between known attacks, which causes them to confidently misclassify unseen/zero-day attacks as "normal" or some known attack type. 

ProtoIDS solves this by learning "prototypes" (representative embeddings) for network behaviors. When an attack comes in, the model measures its cosine distance to these prototypes. If the attack is too far from any known prototype, it is flagged as **UNKNOWN**, rather than being misclassified.

## Project Status
**Current Phase: Merged Implementation Completed**

This repository represents a merged implementation combining the strengths of two ProtoIDS implementations:
- **Repository A**: ProtoIDS-main (original implementation with memory-efficient processing)
- **Repository B**: Capstone Project (friend's implementation with correct open-set protocol)

What has been completed:
- Thorough forensic analysis and preprocessing of two massive datasets: CICIoT2023 and Edge-IIoTset.
- Created a manageable development subset (235,135 samples) of CICIoT2023 for rapid testing.
- Built and evaluated closed-set baselines (Random Forest, Gradient Boosting, MLP).
- Implemented correct open-set protocol with proper unknown class handling.
- Integrated memory-efficient chunked processing with development subset capability.

## Key Features

- **Leakage-Free Preprocessing**: Comprehensive forensic analysis ensures that identifiers like IP addresses, timestamps, and raw payloads are removed, guaranteeing the model learns true behavioral patterns rather than memorizing dataset artifacts.
- **Memory-Efficient Processing**: Implements chunked data processing (100k row chunks) and memmap arrays to handle massive datasets like CICIoT2023 (5.5M+ rows) gracefully without exceeding RAM limits.
- **Rigorous Open-Set Protocol**: Explicitly withholds specific attack classes (e.g., `MITM-ArpSpoofing`, `VulnerabilityScan`, `DictionaryBruteForce`) during training to rigorously evaluate the model's ability to detect novel, zero-day attacks at inference time.
- **Correct Unknown Class Handling**: Prototypes are learned only for known classes; UNKNOWN class is detected via distance-based rejection, not through learned prototypes.
- **Development Subset Capability**: Option to use curated development subset for rapid prototyping while maintaining ability to process full dataset.
- **Cross-Dataset Validation**: Built to evaluate model generalization across completely different environments (CICIoT2023 and Edge-IIoTset).
- **Checkpointing and Metrics**: Training loop saves best model checkpoints and records comprehensive metrics including epoch times and peak memory usage.

## Project Structure

- `src/preprocessing/`: Memory-efficient chunked data processing, feature scaling, and open-set protocol enforcement.
  - `ciciot_preprocessing.py`: For CICIoT2023 dataset
  - `edgeiiot_preprocessing.py`: For Edge-IIoTset dataset
- `src/baselines/`: Scripts to run traditional ML baselines (Random Forest, Gradient Boosting, MLP) and ablation studies.
- `src/protoids/`: Core implementation of the ProtoIDS architecture:
  - `encoder.py`: Feature encoder network
  - `prototype_layer.py`: Learnable prototypes with cosine distance computation
  - `protoids_model.py`: Main ProtoIDS model combining encoder and prototype layers
  - `dataset.py`: Memory-efficient dataset loading with development subset support
  - `training.py`: Training loop with proper unknown class masking and checkpointing
  - `train_protoids.py`: Main training script with open-set evaluation on validation and test sets
  - `autoencoder.py`: Utility for pretraining encoder
  - `pretrain_ae.py`: Script for autoencoder pretraining
  - `cross_evaluate.py`: Utility for cross-dataset evaluation (CICIoT → Edge-IIoT)
- `analysis/`: Threshold analysis tools
- `reports/`: Analysis scripts and comprehensive reports:
  - Forensic analysis reports
  - Methodology documentation
  - Progress summaries
  - Baseline experiment results
- `experiments/`: 
  - `configs/`: Experiment configurations
  - `results/`: Experiment results and baseline references
  - `models/`: Saved model checkpoints
  - `processed/`: Preprocessed data (memmap arrays and development subsets)
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
- `experiments/models/` - for saved model checkpoints  
- `experiments/processed/` - for processed data (memmap arrays)
- `experiments/configs/` - for experiment configurations

### 3. Development Subset
To use the development subset for rapid prototyping, ensure `experiments/data/ciciot_dev.parquet` exists (created by preprocessing script).

### 4. Full Dataset Processing
To process the complete CICIoT2023 dataset (5.5M+ rows), omit the development subset parameter or point to a different subset.

## True Open-Set Protocol

This implementation follows a rigorous open-set protocol:
1. **Withhold Unknown Classes**: During training, specific attack classes are completely withheld from the training data
2. **Fit Scaler on Known Data Only**: StandardScaler is fitted only on known class training samples
3. **Learn Prototypes for Known Classes Only**: Prototype vectors are learned exclusively from known class data
4. **Threshold Calibration**: Distance threshold for UNKNOWN detection is calibrated using validation set known-class distances
5. **Unknown Detection**: Samples with maximum prototype distance exceeding threshold are flagged as UNKNOWN
6. **Validation Separation**: Threshold is calibrated on validation set, then reused for test set evaluation (never recalibrated on test)

## Key Files for Reproduction


## Important Note on Invalid Open-Set Approaches



Any experiment where the ProtoIDS model was trained using all 34 CICIoT2023 classes (including attack classes that should be withheld for open-set evaluation) and then only treated certain classes as UNKNOWN during evaluation must be marked as:



**INVALID FOR FINAL OPEN-SET CLAIMS**



Such approaches do not constitute valid open-set evaluation because:

- The model learns prototype vectors for classes that should remain unknown

- This violates the fundamental open-set assumption that unknown classes are not seen during training

- Results from such experiments may show artificially inflated performance due to leakage



The final research results must use the true open-set protocol where unknown classes are completely withheld from training data, including feature scaling and prototype learning.
- **Preprocessing**: `src/preprocessing/ciciot_preprocessing.py` 
- **Core Model**: `src/protoids/` directory (all files)
- **Training Script**: `src/protoids/train_protoids.py`
- **Analysis**: `analysis/analyze_threshold.py` for threshold sweeping experiments

## References

For detailed forensic analysis, see:
- `reports/dataset_forensic_report.md`
- `reports/protoids_methodology.md`

For baseline results and experimental protocols, see:
- `reports/baseline_experiment_report.md`
- `reports/progress_summary.md`

## Next Steps

1. Run baseline experiments to establish performance benchmarks
2. Execute the full ProtoIDS training pipeline with open-set evaluation
3. Analyze threshold operating points using the provided analysis tools
4. Evaluate cross-dataset generalization on Edge-IIoTset
5. Document findings and identify areas for improvement
