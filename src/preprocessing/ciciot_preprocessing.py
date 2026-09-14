"""
Preprocessing pipeline for CICIoT2023 dataset.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib

def load_ciciot_splits(base_path):
    """
    Load train, validation, and test splits.
    Preserves the original split.
    """
    train_path = os.path.join(base_path, 'train', 'train.csv')
    val_path = os.path.join(base_path, 'validation', 'validation.csv')
    test_path = os.path.join(base_path, 'test', 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df

def get_columns_to_remove():
    """
    Return columns to remove and reasons based on forensic analysis.
    """
    # Target variable (to be separated, not used as feature)
    target_col = ['label']

    # Constant columns (always 0)
    constant_cols = ['Telnet', 'IRC']

    # Near-constant columns (essentially constant) - we remove SMTP as per report
    near_constant_cols = ['SMTP']

    # Redundant pairs: remove one from each pair
    # Rate and Srate are identical -> remove Srate
    redundant_pair1 = ['Srate']  # we keep Rate

    # IPv and LLC are identical (99.99% = 1.0) -> remove LLC
    redundant_pair2 = ['LLC']   # we keep IPv

    cols_to_remove = target_col + constant_cols + near_constant_cols + redundant_pair1 + redundant_pair2
    reasons = {
        'label': 'Target variable',
        'Telnet': 'Constant (always 0)',
        'IRC': 'Constant (always 0)',
        'SMTP': 'Near-constant (essentially constant in validation)',
        'Srate': 'Redundant with Rate (identical values)',
        'LLC': 'Redundant with IPv (identical distributions, 99.99% = 1.0)'
    }

    return cols_to_remove, reasons

def get_feature_columns(df, cols_to_remove):
    """
    Return list of feature column names after removing specified columns.
    """
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in cols_to_remove]
    return feature_cols

def create_label_mappings(train_df):
    """
    Create mappings for original multiclass label and binary label.
    """
    # Original label mapping: string to integer
    label_classes = sorted(train_df['label'].unique())
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    # Binary label: BenignTraffic = 0, everything else = 1
    def binary_label_func(label_str):
        return 0 if label_str == 'BenignTraffic' else 1

    binary_mapping = {
        'BenignTraffic': 0,
        # All other labels map to 1
    }
    # We don't need to list all attack types for binary, just the rule.

    label_mapping = {
        'multiclass': {
            'label_to_int': label_to_int,
            'int_to_label': int_to_label,
            'classes': label_classes
        },
        'binary': {
            'rule': 'BenignTraffic -> 0, else -> 1',
            'BenignTraffic': 0,
            'Attack': 1  # aggregate
        }
    }

    return label_mapping

def preprocess_ciciot(train_df, val_df, test_df, cols_to_remove):
    """
    Preprocess the dataframes:
    - Remove specified columns from features
    - Separate features and labels
    - Fit StandardScaler on training features
    - Transform all splits
    """
    # Get feature columns
    feature_cols = get_feature_columns(train_df, cols_to_remove)

    # Separate features and labels
    X_train = train_df[feature_cols].copy()
    y_train_multi = train_df['label'].copy()

    X_val = val_df[feature_cols].copy()
    y_val_multi = val_df['label'].copy()

    X_test = test_df[feature_cols].copy()
    y_test_multi = test_df['label'].copy()

    # Create binary labels
    y_train_bin = y_train_multi.apply(lambda x: 0 if x == 'BenignTraffic' else 1)
    y_val_bin = y_val_multi.apply(lambda x: 0 if x == 'BenignTraffic' else 1)
    y_test_bin = y_test_multi.apply(lambda x: 0 if x == 'BenignTraffic' else 1)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform validation and test
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to float32 to save memory
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_val_scaled = X_val_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    return (X_train_scaled, y_train_multi, y_train_bin,
            X_val_scaled, y_val_multi, y_val_bin,
            X_test_scaled, y_test_multi, y_test_bin,
            scaler, feature_cols)

def create_development_subset(X_train, y_train_multi, y_train_bin,
                              seed=42, max_medium=5000, max_majority=10000):
    """
    Create a development subset from training data only.
    - Keep all samples from classes with <5,000 training samples
    - Maximum 5,000 samples for medium classes
    - Maximum 10,000 samples for majority classes
    """
    np.random.seed(seed)

    # Get class counts for multi-class labels
    class_counts = y_train_multi.value_counts()

    # Determine which samples to keep
    indices_to_keep = []

    for label_str, count in class_counts.items():
        # Get indices for this class
        class_indices = np.where(y_train_multi.values == label_str)[0]

        if count < 5000:
            # Keep all samples
            indices_to_keep.extend(class_indices)
        elif count < 10000:  # medium class: 5000 <= count < 10000
            # Sample max_medium samples
            if count <= max_medium:
                indices_to_keep.extend(class_indices)
            else:
                sampled = np.random.choice(class_indices, size=max_medium, replace=False)
                indices_to_keep.extend(sampled)
        else:  # majority class: count >= 10000
            # Sample max_majority samples
            if count <= max_majority:
                indices_to_keep.extend(class_indices)
            else:
                sampled = np.random.choice(class_indices, size=max_majority, replace=False)
                indices_to_keep.extend(sampled)

    # Convert to numpy array and sort for consistency
    indices_to_keep = np.sort(np.array(indices_to_keep))

    # Subset the data
    X_train_dev = X_train[indices_to_keep]
    y_train_multi_dev = y_train_multi.iloc[indices_to_keep].reset_index(drop=True)
    y_train_bin_dev = y_train_bin.iloc[indices_to_keep].reset_index(drop=True)

    return X_train_dev, y_train_multi_dev, y_train_bin_dev, indices_to_keep

def save_preprocessing_artifacts(scaler, feature_cols, label_mapping,
                                cols_to_remove, reasons, output_dir):
    """
    Save scaler, feature manifest, and label mapping.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(output_dir, 'ciciot_preprocessor.joblib')
    joblib.dump(scaler, scaler_path)

    # Create feature manifest
    manifest = {
        'original_columns': list(feature_cols) + cols_to_remove,  # all original columns
        'removed_columns': cols_to_remove,
        'retained_columns': feature_cols,
        'removal_reasons': reasons
    }
    manifest_path = os.path.join(output_dir, 'ciciot_feature_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save label mapping
    label_mapping_path = os.path.join(output_dir, 'ciciot_label_mapping.json')
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)

    return scaler_path, manifest_path, label_mapping_path

def main():
    """
    Main preprocessing pipeline for CICIoT2023.
    """
    print("Starting CICIoT2023 preprocessing...")

    # Paths - relative to project root
    data_base_path = 'CICIOT23'
    output_dir = 'results/baselines'

    # Load data
    print("Loading data splits...")
    train_df, val_df, test_df = load_ciciot_splits(data_base_path)
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # Get columns to remove
    cols_to_remove, reasons = get_columns_to_remove()
    print(f"Columns to remove: {cols_to_remove}")
    print(f"Reasons: {reasons}")

    # Preprocess
    print("Preprocessing data...")
    (X_train_scaled, y_train_multi, y_train_bin,
     X_val_scaled, y_val_multi, y_val_bin,
     X_test_scaled, y_test_multi, y_test_bin,
     scaler, feature_cols) = preprocess_ciciot(train_df, val_df, test_df, cols_to_remove)

    print(f"Features after removal: {len(feature_cols)}")
    print(f"Train features shape: {X_train_scaled.shape}")

    # Create label mappings
    print("Creating label mappings...")
    label_mapping = create_label_mappings(train_df)

    # Create development subset
    print("Creating development subset...")
    X_train_dev, y_train_multi_dev, y_train_bin_dev, dev_indices = create_development_subset(
        X_train_scaled, y_train_multi, y_train_bin
    )
    print(f"Development subset size: {X_train_dev.shape[0]} samples")

    # Save preprocessing artifacts
    print("Saving preprocessing artifacts...")
    scaler_path, manifest_path, label_mapping_path = save_preprocessing_artifacts(
        scaler, feature_cols, label_mapping, cols_to_remove, reasons, output_dir
    )

    # Save processed data (optional, but useful for experiments)
    processed_dir = 'experiments/data'
    os.makedirs(processed_dir, exist_ok=True)

    # Save development subset as Parquet for efficiency
    # We'll save features and labels together in a DataFrame for convenience
    dev_df = pd.DataFrame(X_train_dev, columns=feature_cols)
    dev_df['label_multi'] = y_train_multi_dev.values
    dev_df['label_binary'] = y_train_bin_dev.values
    dev_parquet_path = os.path.join(processed_dir, 'ciciot_dev.parquet')
    dev_df.to_parquet(dev_parquet_path, index=False)

    # Save development metadata
    dev_metadata = {
        'seed': 42,
        'source': 'CICIOT23 training data',
        'sampling_rules': {
            'minority_classes': '<5000 samples -> keep all',
            'medium_classes': '5000 <= count < 10000 -> max 5000 samples',
            'majority_classes': '>=10000 samples -> max 10000 samples'
        },
        'total_rows': int(X_train_dev.shape[0]),
        'class_counts': y_train_multi_dev.value_counts().to_dict()
    }
    dev_metadata_path = os.path.join(processed_dir, 'ciciot_dev_metadata.json')
    with open(dev_metadata_path, 'w') as f:
        json.dump(dev_metadata, f, indent=2)

    # Also save the full processed splits if needed (optional)
    # For now, we just save the development subset and the preprocessing objects.
    print("Preprocessing complete!")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved feature manifest to: {manifest_path}")
    print(f"Saved label mapping to: {label_mapping_path}")
    print(f"Saved development subset to: {dev_parquet_path}")
    print(f"Saved development metadata to: {dev_metadata_path}")

if __name__ == '__main__':
    main()