# Next Person's Guide: Working with X-IIoTID

This document provides specific instructions for the next researcher who will work with the X-IIoTID dataset for the ProtoIDS project.

## Overview
The X-IIoTID dataset is the third dataset that needs to be integrated into the ProtoIDS framework. This guide outlines the steps required to:
1. Obtain and inspect the dataset
2. Audit for leakage and analyze characteristics
3. Define true-open-set classes with justification
4. Build dataset-specific preprocessing
5. Train and evaluate ProtoIDS with proper open-set protocol
6. Document results and limitations

## Step 1: Obtain the X-IIoTID Dataset

### Acquisition
- **Source**: Obtain X-IIoTID from an authoritative source (to be determined)
- **Common Sources**: 
  - Academic repositories (IEEE DataPort, Kaggle, UCI ML Repository)
  - Direct from authors if available through research collaboration
  - Institutional access through university subscriptions
- **Version**: Note the exact version and date of acquisition
- **License**: Check usage rights and any restrictions

### Initial Setup
```bash
# Create directory for X-IIoTID data
mkdir -p datasets/xiiotid
# Place downloaded files in this directory
# DO NOT commit large datasets to git (they are ignored via .gitignore)
```

## Step 2: Initial Inspection and Schema Analysis

### File Examination
- List all files and their sizes
- Examine file formats (CSV, ARFF, etc.)
- Check for README or documentation files

### Basic Statistics
- Number of instances (rows)
- Number of features (columns)
- Feature types (numerical, categorical, etc.)
- Presence of labels/columns indicating attack types
- Timestamp, IP address, and port columns (potential leakage sources)

### Sample Command Script
```python
# Example Python script for initial inspection
import pandas as pd
import numpy as np

def inspect_xiiotid(filepath):
    print(f"Inspecting: {filepath}")
    
    # Load data (adjust based on format)
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.arff'):
        from scipy.io import arff
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
    else:
        raise ValueError("Unsupported file format")
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # If label column exists
    if 'label' in df.columns or 'attack_type' in df.columns or 'class' in df.columns:
        label_col = [c for c in ['label', 'attack_type', 'class'] if c in df.columns][0]
        print(f"\nLabel distribution:\n{df[label_col].value_counts()}")
    
    return df

# Usage
# df = inspect_xiiotid("datasets/xiiotid/[filename]")
```

## Step 3: Leakage Analysis and Feature Selection

### Leakage-Prone Columns to Remove
Based on lessons from CICIoT2023 and Edge-IIoTset, identify and remove:
- **Identifiers**: IP addresses (source/destination), MAC addresses
- **Temporal markers**: Timestamps, sequence numbers, flow start/end times
- **Port numbers**: Source and destination ports (may encode protocol-specific behavior)
- **Packet counts/bytes**: May correlate directly with attack volume rather than behavior
- **Protocol-specific fields**: That may leak implementation details
- **Any column that uniquely identifies a flow or session**

### Feature Selection Process
1. **Correlation Analysis**: Identify highly correlated features
2. **Variance Threshold**: Remove near-constant features
3. **Domain Knowledge**: Consult with cybersecurity experts on feature relevance
4. **Leakage Check**: Ensure no identifiers or temporal leakage remains
5. **Final Set**: Aim for 30-50 numerical features (similar to prior datasets)

### Documentation
- Create a feature selection report in `docs/experiments/X-IIoTID/`
- List all removed columns with justification
- Document the final feature set and rationale

## Step 4: Define True-Open-Set Classes

### Withholding Strategy
Follow the established ProtoIDS protocol:
1. **Withhold specific attack classes completely from training**
2. **Ensure these classes appear only in validation and test sets**
3. **Never use unknown class data for training, scaling, or prototype learning**

### Class Selection Criteria
Select withheld classes based on:
1. **Zero-Day Relevance**: Classes representing attack types likely to appear as novel threats
2. **Sample Sufficiency**: Enough instances in validation/test for reliable statistics
3. **Distinctiveness**: Represent different attack categories or behaviors
4. **Balance**: Avoid withholding extremely rare classes that would yield unreliable metrics

### Recommended Approach
- **Consult Domain Experts**: If available, get input on which classes are most relevant
- **Analyze Attack Categories**: Group similar attacks and select representatives
- **Check Temporal/Spatial Patterns**: Ensure withheld classes are not artifacts of specific time periods
- **Document Justification**: Clearly explain why each selected class was chosen for withholding

### Example withheld classes (to be determined after inspection):
- [To be filled after dataset inspection]
- **Justification**: [To be filled]

### Validation
- Verify that withheld classes have zero instances in training data
- Confirm that scaling parameters are computed only on known training data
- Ensure validation set contains both known and unknown samples

## Step 5: Build Dataset-Specific Preprocessing

### Preprocessing Pipeline Requirements
Create a preprocessing script similar to:
- `src/preprocessing/ciciot_preprocessing.py`
- `src/preprocessing/edgeiiot_preprocessing.py`

### Essential Components
1. **Data Loading**: Read and parse the X-IIoTID dataset format
2. **Leakage Removal**: Drop identified leakage-prone columns
3. **Feature Extraction**: Convert to numerical features if needed
4. **Missing Value Handling**: 
   - Strategy: Mean/median imputation for numerical features
   - Documentation: Justify chosen strategy
   - Alternative: Remove rows/columns with excessive missingness (document threshold)
5. **Outlier Handling** (Optional):
   - Winsorizing or clipping at extreme percentiles
   - Document if applied and at what levels
6. **Feature Scaling**:
   - **Critical**: Fit StandardScaler ONLY on known training data (after withholding)
   - Save scaler for reuse in validation/test transformation
   - Do NOT fit on validation or test data
7. **Train/Validation/Test Split**:
   - Use existing splits if provided
   - If creating splits: stratify by class, ensure unknown classes only in validation/test
   - Document split ratios and stratification

### Implementation Template
```python
# Example structure for X-IIoTID preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_xiiotid(
    raw_data_path,
    output_dir,
    withheld_classes=None,  # List of class labels to withhold
    test_size=0.2,
    val_size=0.1,
    random_state=42
):
    """
    Preprocess X-IIoTID dataset for ProtoIDS.
    
    Args:
        raw_data_path: Path to raw X-IIoTID dataset
        output_dir: Directory to save preprocessed data
        withheld_classes: List of class labels to completely withhold from training
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with paths to preprocessed data splits
    """
    # 1. Load raw data
    df = load_raw_data(raw_data_path)
    
    # 2. Identify label column
    label_col = identify_label_column(df)
    
    # 3. Withhold specified classes (BEFORE any processing)
    if withheld_classes is not None:
        known_mask = ~df[label_col].isin(withheld_classes)
        unknown_mask = df[label_col].isin(withheld_classes)
        
        df_known = df[known_mask].copy()
        df_unknown = df[unknown_mask].copy()
        
        print(f"Withheld {len(withheld_classes)} classes: {withheld_classes}")
        print(f"Known samples: {len(df_known)}")
        print(f"Unknown samples: {len(df_unknown)}")
    else:
        df_known = df.copy()
        df_unknown = pd.DataFrame(columns=df.columns)  # Empty
    
    # 4. Remove leakage-prone columns
    columns_to_drop = identify_leakage_columns(df_known)
    df_known_dropped = df_known.drop(columns=columns_to_drop)
    df_unknown_dropped = df_unknown.drop(columns=columns_to_drop) if not df_unknown.empty else df_unknown
    
    # 5. Handle missing values
    df_known_processed = handle_missing_values(df_known_dropped)
    df_unknown_processed = handle_missing_values(df_unknown_dropped)
    
    # 6. Feature extraction (if needed)
    # Example: convert categorical to numerical, create aggregations, etc.
    
    # 7. Ensure only numerical features remain
    feature_cols = get_numerical_feature_columns(df_known_processed)
    
    # 8. Fit scaler ONLY on known training data
    # Split known data into train and val (for threshold calibration)
    X_known = df_known_processed[feature_cols].values
    y_known = df_known_processed[label_col].values
    
    # Split known data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_known, y_known, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_known
    )
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Transform unknown data (if any) using same scaler
    if not df_unknown_processed.empty:
        X_unknown = df_unknown_processed[feature_cols].values
        X_unknown_scaled = scaler.transform(X_unknown)
    else:
        X_unknown_scaled = np.empty((0, len(feature_cols)))
    
    # 9. Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_unknown_scaled)  # Unknown test
    np.save(os.path.join(output_dir, 'y_test.npy'), 
            np.array([-1] * len(X_unknown_scaled)) if not df_unknown_processed.empty else np.array([]))  # Unknown label
    
    # Save scaler and feature info
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # Save withheld classes info
    with open(os.path.join(output_dir, 'withheld_classes.txt'), 'w') as f:
        f.write('\n'.join(withheld_classes) if withheld_classes else '')
    
    return {
        'output_dir': output_dir,
        'feature_names': feature_cols,
        'withheld_classes': withheld_classes,
        'scaler_path': os.path.join(output_dir, 'scaler.joblib')
    }

# Helper functions to be implemented based on dataset inspection
def load_raw_data(filepath): ...
def identify_label_column(df): ...
def identify_leakage_columns(df): ...
def handle_missing_values(df): ...
def get_numerical_feature_columns(df): ...
```

## Step 6: Train and Evaluate ProtoIDS

### Training Commands
Once preprocessing is complete and data is saved in NumPy format:

```bash
# Update the dataset.py to handle X-IIoTID format (or create a new variant)
# For simplicity, assume we adapt the existing dataset loader

# Closed-set experiment (development subset if created)
python src/protoids/train_protoids.py --experiment_name protoids_xiiotid_closed_dev

# Open-set experiment (with proper withholding)
python src/protoids/train_protoids.py --withhold_open_set --experiment_name protoids_xiiotid_open_dev

# Full data experiments (if memory-efficient implementation ready)
python src/protoids/train_protoids.py --withhold_open_set --full_data --experiment_name protoids_xiiotid_open_full
```

### Key Checks During Training
- Verify that withheld classes are absent from training data
- Confirm scaler is fitted only on known training data
- Ensure threshold is computed only on known validation samples
- Check that unknown test classes never influence training or threshold

### Expected Outputs
- Model checkpoints: `experiments/models/protoids_xiiotid_*/`
- Results: `experiments/results/protoids_xiiotid_*/`
- Should include:
  - Training history
  - Closed-set results (accuracy, macro F1, etc.)
  - Open-set results (known/unknown precision, recall, F1, AUROC, AUPR)
  - Threshold value used

## Step 7: Threshold Analysis and Result Documentation

### Threshold Sensitivity Analysis
- Test different threshold percentiles (80th, 85th, 90th, 95th) on validation set
- Analyze impact on known-class accuracy and unknown recall
- Select optimal threshold based on application priorities (or use 90th as default)
- Document analysis in `docs/experiments/X-IIoTID/threshold_analysis.md`

### Result Documentation
Create a results summary similar to those in `docs/experiments/CICIoT2023/` and `docs/experiments/Edge-IIoTset/`:

**Dataset: X-IIoTID**
- Known Classes: [number] (withheld: [list])
- Known Training Samples: [number]
- Unknown Validation: [number] 
- Unknown Test: [number]
- Input Features: [number]
- Embedding: [dimension]-dimensional L2-normalized
- Prototypes: [number] per known class
- [Other hyperparameters: dropout, learning rate, lambda, batch, epochs, seed]

**Test Results:**
- [Insert metrics table similar to previous datasets]

### Artifact Preservation
- Copy result JSON files to `results/documented/X-IIoTID/`
- Save plots and visualizations to `docs/experiments/X-IIoTID/`
- Archive preprocessing scripts and parameters

## Step 8: Cross-Dataset Validation (Future Work)
After successful X-IIoTID processing:
1. **Train on CICIoT2023, test on X-IIoTID** (and vice versa)
2. **Train on Edge-IIoTset, test on X-IIoTID** (and vice versa)
3. **Analyze performance drop** and investigate causes
4. **Consider feature alignment techniques** if needed
5. **Document findings** in cross-dataset validation report

## Limitations and Assumptions
Document all assumptions made during the process:
- Assumptions about feature meaning and relevance
- Decisions regarding missing value handling
- Outlier treatment (if any)
- Withheld class justification
- Preprocessing protocol adherence
- Computational constraints encountered

## Resources and References
- **ProtoIDS Preprocessing Examples**:
  - `src/preprocessing/ciciot_preprocessing.py`
  - `src/preprocessing/edgeiiot_preprocessing.py`
- **Dataset Forensic Reports**:
  - `reports/dataset_forensic_report.md`
  - `docs/ARTIFACTS_AND_BACKUPS.md`
- **Reproducibility Guidelines**:
  - `docs/REPRODUCIBILITY.md`
- **Research Decisions**:
  - `docs/RESEARCH_DECISIONS.md`
- **Leakage Audit Checklist**:
  - See `HANDOVER.md` for detailed checklist

## Important Notes
1. **Do NOT Commit Large Files**: Raw datasets, intermediate CSV files, and large NumPy arrays must remain in `datasets/` or `experiments/` directories (which are ignored by git)
2. **Preserve Protocol**: Never compromise the true open-set protocol (withholding → fitting → transforming) for convenience
3. **Validate Leakage**: Always double-check that no unknown class information leaks into training
4. **Seek Guidance**: If unsure about any step, consult the project documentation or available mentors
5. **Iterate Gradually**: Get the development subset working first before attempting full dataset

---
*Next steps after X-IIoTID: Cross-dataset validation Studies and Final Report Preparation*