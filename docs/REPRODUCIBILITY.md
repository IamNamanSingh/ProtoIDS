# Reproducibility Guidelines

This document outlines the steps to ensure reproducibility of ProtoIDS experiments and results.

## Environment Setup
1. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   cd ProtoIDS-merged
   ```
2. **Checkout the Correct Branch**:
   ```bash
   git checkout protoids-next
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify Environment**:
   - Python version: 3.8+
   - Key packages: torch, torchvision, scikit-learn, numpy, pandas, joblib

## Data Preparation
### Development Subset (for rapid experimentation)
- The development subset is automatically created by the preprocessing script.
- Location: `experiments/data/ciciot_dev.parquet` (~235k rows)
- To regenerate: Run the CICIoT2023 preprocessing script with development subset flag
- **Note**: The development subset is sufficient for verifying code correctness and initial experiments.

### Full Dataset Processing
- Full CICIoT2023 dataset (~5.5M rows) requires memory-efficient implementation.
- Location: Point to the full CSV dataset in the preprocessing script.
- **Important**: The full dataset is NOT included in the repository due to size constraints.
- Obtain CICIoT2023 from: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/icict-2023.html)

## Experiment Execution
### Setting Seeds for Reproducibility
All experiments should set random seeds for reproducibility:
- Python random seed: `random.seed(42)`
- NumPy random seed: `np.random.seed(42)`
- PyTorch random seed: `torch.manual_seed(42)`
- If using CUDA: `torch.cuda.manual_seed_all(42)`

### ProtoIDS Training Commands
Use the following commands to run experiments:

**Closed-set experiment (development subset)**:
```bash
python src/protoids/train_protoids.py --experiment_name protoids_closed_dev
```

**Open-set experiment (development subset)**:
```bash
python src/protoids/train_protoids.py --withhold_open_set --experiment_name protoids_open_dev
```

**Full data experiments (require memory-efficient implementation)**:
```bash
python src/protoids/train_protoids.py --withhold_open_set --full_data --experiment_name protoids_open_full
```

### Key Flags
- `--withhold_open_set`: Enables true open-set mode (withholds attack classes from training)
- `--full_data`: Uses full 5.5M CICIoT2023 training set (not development subset)
- `--experiment_name`: Name for saving results and models (used in `experiments/results/`)
- `--seed`: Random seed for reproducibility (default 42)

## Verification Steps
To verify that an experiment is reproducible:

1. **Checkpoint and Scaler Location**:
   - Models and scalers are saved in: `experiments/models/[experiment_name]/`
   - Check for: `model.pth`, `best_model.pth`, `prototypes.npy`, `threshold.npy`, `label_mapping.json`, `preprocessor.joblib`

2. **Results Location**:
   - Experiment logs and metrics are saved in: `experiments/results/[experiment_name]/`
   - Check for: `training_history.json`, `closed_set_results.json`, `open_set_results.json`, `open_set_test_results.json`

3. **Result Validation**:
   - Compare key metrics (accuracy, macro F1, etc.) with expected values from documented results
   - Allow for minor floating-point differences due to hardware/platform variations
   - Significant differences may indicate protocol deviations

4. **Leakage Audit Verification**:
   - Ensure preprocessing follows: withhold first → fit scaler only on known training data → transform validation/test using same scaler
   - Check logs or add assertions to confirm:
     - No unknown class data in training set after withholding
     - Scaler fitted only on known training data
     - Threshold calculated only on known validation samples

## Troubleshooting
### Common Issues
1. **Out of Memory Errors**:
   - Reduce batch size (`--batch_size` argument)
   - Ensure memory-efficient preprocessing is implemented for full data
   - Use development subset for initial testing

2. **Module Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and virtual environment

3. **File Not Found Errors**:
   - Verify data paths in configuration or command-line arguments
   - Ensure development subset exists or full data path is correct

4. **Non-Deterministic Results**:
   - Confirm all random seeds are set (Python, NumPy, PyTorch)
   - Check for non-deterministic operations (e.g., some GPU operations)
   - Use CPU for debugging if necessary

## Reproducibility Notes
- **Hardware Variations**: Results may vary slightly between CPU/GPU and different hardware due to floating-point precision.
- **Software Versions**: Try to match the dependency versions in `requirements.txt` for exact reproducibility.
- **Data Variants**: Ensure you are using the exact same dataset splits (train/validation/test) as used in the original experiments.
- **Protocol Adherence**: Deviations from the true open-set protocol (withholding → fitting → transforming) will invalidate results for open-set claims.

## References
- For detailed preprocessing protocols, see:
  - `src/preprocessing/ciciot_preprocessing.py`
  - `src/preprocessing/edgeiiot_preprocessing.py`
- For training configuration, see:
  - `src/protoids/train_protoids.py`
  - `src/protoids/training.py`
- For dataset splits and protocol, see:
  - `reports/dataset_forensic_report.md`
  - `docs/ARTIFACTS_AND_BACKUPS.md`