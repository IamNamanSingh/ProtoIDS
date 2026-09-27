"""
Dataset creation utilities for ProtoIDS.
Handles loading, preprocessing, and creating data loaders for CICIoT2023.
"""

import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Optional

class ParquetDataset(Dataset):
    """Dataset that loads data from a Parquet file (already preprocessed)."""
    def __init__(self, parquet_path: str, feature_cols: list, label_to_int: dict):
        # Load the Parquet file
        self.df = pd.read_parquet(parquet_path)
        
        # Extract features and labels
        self.feature_cols = feature_cols
        self.label_to_int = label_to_int
        self.unknown_idx = label_to_int.get('UNKNOWN', -1)
        
        # Prepare features (already scaled, just need to clip)
        self.X = self.df[feature_cols].values.astype(np.float32)
        self.X = np.clip(self.X, -5, 5)  # Clip to [-5, 5] as in original preprocessing
        
        # Prepare labels
        y_multi_raw = self.df['label'].values
        self.y = np.vectorize(lambda x: label_to_int.get(x, self.unknown_idx))(y_multi_raw)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Return features and label as torch tensors
        features = torch.from_numpy(self.X[idx]).float()
        label = torch.as_tensor(self.y[idx], dtype=torch.long)
        return features, label
    
    def get_num_classes(self):
        return len(self.label_to_int)
    
    def get_input_dim(self):
        return len(self.feature_cols)


class MemmapDataset(Dataset):
    """Dataset that wraps memmap arrays for efficient large-scale data loading."""
    def __init__(self, features_path, labels_path, num_samples, input_dim, unknown_class_index=None, label_mapping=None):
        self.features = np.memmap(features_path, dtype='float32', mode='r', shape=(num_samples, input_dim))
        self.labels = np.memmap(labels_path, dtype='int64', mode='r', shape=(num_samples,))
        self.input_dim = input_dim
        self.unknown_class_index = unknown_class_index
        self.label_mapping = label_mapping  # store new_label_to_int for reference

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Return features and label as torch tensors
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return features, label

    def get_num_classes(self):
        # For compatibility, return number of known classes (excluding unknown)
        if self.unknown_class_index is not None:
            return self.unknown_class_index  # number of known classes
        else:
            # fallback: should not happen in open_set=True
            return len(self.label_mapping) if self.label_mapping else -1

    def get_input_dim(self):
        return self.input_dim

    def get_unknown_class_index(self):
        return self.unknown_class_index


def create_data_loaders(data_dir: str, batch_size: int = 256,
                        artifact_dir: str = 'results/baselines',
                        development_subset_path: Optional[str] = None,
                        withheld_classes: Optional[list] = None,
                        open_set: bool = False,
                        scaler_save_dir: Optional[str] = None):
    """
    Create data loaders for training, validation, and test sets.
    Handles both closed-set and open-set scenarios.
    Implements memory-efficient processing for large datasets.

    Args:
        data_dir: Path to CICIoT2023 dataset directory
        batch_size: Batch size for data loaders
        artifact_dir: Directory containing feature and label mappings
        development_subset_path: Path to Parquet file for development subset (if None, use full dataset)
        withheld_classes: List of class indices to withhold from training (for open-set)
        open_set: Whether to perform open-set processing (withhold classes before scaling)
        scaler_save_dir: Directory to save scaler and label mappings (if open_set)

    Returns:
        train_loader, val_loader, test_loader, num_classes, input_dim
    """
    print(f"Loading feature manifest from {artifact_dir}")
    with open(os.path.join(artifact_dir, 'ciciot_feature_manifest.json'), 'r') as f:
        manifest_orig = json.load(f)

    print(f"Loading original label mapping from {artifact_dir}")
    with open(os.path.join(artifact_dir, 'ciciot_label_mapping.json'), 'r') as f:
        label_mapping_orig = json.load(f)

    feature_cols = manifest_orig['retained_columns']

    # Paths to CSV files
    train_csv_path = os.path.join(data_dir, 'train', 'train.csv')
    val_csv_path = os.path.join(data_dir, 'validation', 'validation.csv')
    test_csv_path = os.path.join(data_dir, 'test', 'test.csv')

    # We will create a directory for processed data under experiments/processed/<experiment_name>
    # Use scaler_save_dir to determine the experiment name (replace 'results' with 'processed')
    if scaler_save_dir is not None:
        processed_base_dir = scaler_save_dir.replace('results', 'processed')
    else:
        processed_base_dir = os.path.join('experiments', 'processed', 'tmp')
    os.makedirs(processed_base_dir, exist_ok=True)
    print(f"Processed data will be saved to: {processed_base_dir}")

    # ----- PASS 1: TRAINING - FIT SCALER AND COLLECT KNOWN LABELS -----
    print("Pass 1: Fitting scaler on training data (after withholding) and collecting known labels...")
    # Initialize scaler
    scaler = StandardScaler()

    # First pass: read training data in chunks to fit scaler and collect known labels
    known_labels_set = set()
    num_training_samples = 0
    chunk_size = 100000  # Process 100k rows at a time

    # Determine which labels are known (not withheld)
    if withheld_classes is not None and len(withheld_classes) > 0:
        known_mask_func = lambda x: (x not in withheld_classes) and (x != -1)
    else:
        known_mask_func = lambda x: x != -1

    # Read train.csv in chunks
    for chunk_df in pd.read_csv(train_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        # Extract features and labels
        X_chunk = chunk_df[feature_cols].values.astype(np.float32)
        y_chunk_raw = chunk_df['label'].values

        # Map string labels to original integer indices
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)

        # Create mask: keep if label is not in withheld_classes and not -1 (invalid)
        mask = np.array([known_mask_func(x) for x in y_chunk_orig])

        if np.any(mask):
            # Fit scaler on known samples
            scaler.partial_fit(X_chunk[mask])
            # Collect known labels (original indices)
            known_labels_set.update(y_chunk_orig[mask])
            num_training_samples += np.sum(mask)

    # Determine number of known classes (excluding withheld)
    num_known_classes = len(known_labels_set)
    print(f"  - Number of known classes in training (after withholding): {num_known_classes}")
    print(f"  - Number of training samples (after withholding): {num_training_samples}")

    # Determine unknown class index (if open_set)
    if open_set:
        # Unknown class index is after all known classes
        unknown_class_index = num_known_classes
        # Total classes = known + 1 (unknown)
        num_classes = num_known_classes + 1
        print(f"  - Unknown class index: {unknown_class_index}")
        print(f"  - Total classes (including unknown): {num_classes}")
    else:
        # Closed-set: all classes are known
        unknown_class_index = None
        # Get total number of classes from original label mapping
        num_classes = len(label_mapping_orig['multiclass']['label_to_int'])
        print(f"  - Total classes (closed-set): {num_classes}")

    # Save scaler and label mappings (known classes only) if scaler_save_dir provided
    if scaler_save_dir is not None:
        print(f"Saving scaler and label mappings to {scaler_save_dir}")
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_save_dir, 'ciciot_preprocessor.joblib'))
        # Create new label mapping: known classes only (0 to num_known_classes-1)
        new_label_to_int = {}
        # Sort known labels for deterministic mapping
        sorted_known_labels = sorted(list(known_labels_set))
        for new_idx, orig_label in enumerate(sorted_known_labels):
            new_label_to_int[int(orig_label)] = new_idx
        with open(os.path.join(scaler_save_dir, 'ciciot_label_mapping.json'), 'w') as f:
            json.dump(new_label_to_int, f, indent=2)
        print(f"DEBUG: About to save label mapping to {os.path.join(scaler_save_dir, 'ciciot_label_mapping.json')}")
        print(f"DEBUG: Label mapping saved")

    # ----- PASS 2: TRAINING - CREATE AND SAVE PROCESSED DATA TO MEMMAP ARRAYS -----
    print("Pass 2: Creating processed training data and saving to memmap arrays...")
    processed_train_dir = os.path.join(processed_base_dir, 'train')
    os.makedirs(processed_train_dir, exist_ok=True)
    train_features_path = os.path.join(processed_train_dir, 'features.dat')
    train_labels_path = os.path.join(processed_train_dir, 'labels.dat')

    # Remove existing files if any to avoid conflicts
    try:
        if os.path.exists(train_features_path):
            os.remove(train_features_path)
    except PermissionError:
        # If file is locked, try to truncate it
        with open(train_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(train_labels_path):
            os.remove(train_labels_path)
    except PermissionError:
        with open(train_labels_path, 'r+b') as f:
            f.truncate(0)

    # Create memmap arrays for training data
    train_features = np.memmap(train_features_path, dtype='float32', mode='w+', shape=(num_training_samples, len(feature_cols)))
    train_labels = np.memmap(train_labels_path, dtype='int64', mode='w+', shape=(num_training_samples,))

    current_idx = 0
    # Read train.csv again in chunks
    for chunk_df in pd.read_csv(train_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        # Extract features and labels
        X_chunk = chunk_df[feature_cols].values.astype(np.float32)
        y_chunk_raw = chunk_df['label'].values

        # Map string labels to original integer indices
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)

        # Create mask: keep if label is not in withheld_classes and not -1 (invalid)
        if withheld_classes is not None and len(withheld_classes) > 0:
            mask = (~np.isin(y_chunk_orig, withheld_classes)) & (y_chunk_orig != -1)
        else:
            mask = (y_chunk_orig != -1)

        if np.any(mask):
            # Transform features with fitted scaler
            X_chunk_transformed = scaler.transform(X_chunk[mask])
            # Clip to [-5, 5] as per original preprocessing
            X_chunk_transformed = np.clip(X_chunk_transformed, -5, 5)
            # Store in memmap
            chunk_size_actual = np.sum(mask)
            train_features[current_idx:current_idx + chunk_size_actual] = X_chunk_transformed
            # Map training labels: known classes to new indices
            if scaler_save_dir is not None and new_label_to_int is not None:
                # Training labels should only contain known classes (since we withheld unknown)
                mapped_labels = np.vectorize(lambda x: new_label_to_int[int(x)])(y_chunk_orig[mask])
                train_labels[current_idx:current_idx + chunk_size_actual] = mapped_labels
            else:
                # Closed-set: keep original labels
                train_labels[current_idx:current_idx + chunk_size_actual] = y_chunk_orig[mask]
            current_idx += chunk_size_actual

    # Flush to ensure data is written
    train_features.flush()
    train_labels.flush()

    # ----- PASS 3: VALIDATION - PROCESS AND SAVE TO MEMMAP -----
    print("Pass 3: Processing validation data and saving to memmap arrays...")
    processed_val_dir = os.path.join(processed_base_dir, 'validation')
    os.makedirs(processed_val_dir, exist_ok=True)
    val_features_path = os.path.join(processed_val_dir, 'features.dat')
    val_labels_path = os.path.join(processed_val_dir, 'labels.dat')

    # Remove existing files if any
    try:
        if os.path.exists(val_features_path):
            os.remove(val_features_path)
    except PermissionError:
        # If file is locked, try to truncate it
        with open(val_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(val_labels_path):
            os.remove(val_labels_path)
    except PermissionError:
        with open(val_labels_path, 'r+b') as f:
            f.truncate(0)

    # First, count validation samples
    num_val_samples = 0
    for chunk_df in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        y_chunk_raw = chunk_df['label'].values
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)
        # For validation, keep all valid labels (including withheld classes)
        mask = (y_chunk_orig != -1)
        num_val_samples += np.sum(mask)

    # Create memmap arrays for validation data
    val_features = np.memmap(val_features_path, dtype='float32', mode='w+', shape=(num_val_samples, len(feature_cols)))
    val_labels = np.memmap(val_labels_path, dtype='int64', mode='w+', shape=(num_val_samples,))

    current_idx = 0
    # Read val.csv again in chunks
    for chunk_df in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        # Extract features and labels
        X_chunk = chunk_df[feature_cols].values.astype(np.float32)
        y_chunk_raw = chunk_df['label'].values

        # Map string labels to original integer indices
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)

        # For validation, keep all valid labels (including withheld classes)
        mask = (y_chunk_orig != -1)

        if np.any(mask):
            # Transform features with fitted scaler
            X_chunk_transformed = scaler.transform(X_chunk[mask])
            # Clip to [-5, 5] as per original preprocessing
            X_chunk_transformed = np.clip(X_chunk_transformed, -5, 5)
            # Store in memmap
            chunk_size_actual = np.sum(mask)
            val_features[current_idx:current_idx + chunk_size_actual] = X_chunk_transformed
            # Map validation labels: known classes to new indices, withheld to unknown
            if scaler_save_dir is not None and new_label_to_int is not None:
                # Map validation labels: known classes to new indices, withheld to unknown
                def map_label(x):
                    xi = int(x)
                    if withheld_classes is not None and xi in withheld_classes:
                        return unknown_class_index
                    else:
                        return new_label_to_int.get(xi, xi)  # fallback to original if not in mapping (should not happen)
                mapped_labels = np.vectorize(map_label)(y_chunk_orig[mask])
                val_labels[current_idx:current_idx + chunk_size_actual] = mapped_labels
            else:
                # Closed-set: keep original labels
                val_labels[current_idx:current_idx + chunk_size_actual] = y_chunk_orig[mask]
            current_idx += chunk_size_actual

    # Flush to ensure data is written
    val_features.flush()
    val_labels.flush()

    # ----- PASS 4: TEST - PROCESS AND SAVE TO MEMMAP -----
    print("Pass 4: Processing test data and saving to memmap arrays...")
    processed_test_dir = os.path.join(processed_base_dir, 'test')
    os.makedirs(processed_test_dir, exist_ok=True)
    test_features_path = os.path.join(processed_test_dir, 'features.dat')
    test_labels_path = os.path.join(processed_test_dir, 'labels.dat')

    # Remove existing files if any
    try:
        if os.path.exists(test_features_path):
            os.remove(test_features_path)
    except PermissionError:
        # If file is locked, try to truncate it
        with open(test_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(test_labels_path):
            os.remove(test_labels_path)
    except PermissionError:
        with open(test_labels_path, 'r+b') as f:
            f.truncate(0)

    # First, count test samples
    num_test_samples = 0
    for chunk_df in pd.read_csv(test_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        y_chunk_raw = chunk_df['label'].values
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)
        # For test, keep all valid labels (including withheld classes)
        mask = (y_chunk_orig != -1)
        num_test_samples += np.sum(mask)

    # Create memmap arrays for test data
    test_features = np.memmap(test_features_path, dtype='float32', mode='w+', shape=(num_test_samples, len(feature_cols)))
    test_labels = np.memmap(test_labels_path, dtype='int64', mode='w+', shape=(num_test_samples,))

    current_idx = 0
    # Read test.csv again in chunks
    for chunk_df in pd.read_csv(test_csv_path, chunksize=chunk_size, usecols=feature_cols + ['label']):
        # Extract features and labels
        X_chunk = chunk_df[feature_cols].values.astype(np.float32)
        y_chunk_raw = chunk_df['label'].values

        # Map string labels to original integer indices
        y_chunk_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_chunk_raw)

        # For test, keep all valid labels (including withheld classes)
        mask = (y_chunk_orig != -1)

        if np.any(mask):
            # Transform features with fitted scaler
            X_chunk_transformed = scaler.transform(X_chunk[mask])
            # Clip to [-5, 5] as per original preprocessing
            X_chunk_transformed = np.clip(X_chunk_transformed, -5, 5)
            # Store in memmap
            chunk_size_actual = np.sum(mask)
            test_features[current_idx:current_idx + chunk_size_actual] = X_chunk_transformed
            # Map test labels: known classes to new indices, withheld to unknown
            if scaler_save_dir is not None and new_label_to_int is not None:
                # Map test labels: known classes to new indices, withheld to unknown
                def map_label(x):
                    xi = int(x)
                    if withheld_classes is not None and xi in withheld_classes:
                        return unknown_class_index
                    else:
                        return new_label_to_int.get(xi, xi)  # fallback to original if not in mapping (should not happen)
                mapped_labels = np.vectorize(map_label)(y_chunk_orig[mask])
                test_labels[current_idx:current_idx + chunk_size_actual] = mapped_labels
            else:
                # Closed-set: keep original labels
                test_labels[current_idx:current_idx + chunk_size_actual] = y_chunk_orig[mask]
            current_idx += chunk_size_actual

    # Flush to ensure data is written
    test_features.flush()
    test_labels.flush()

    # Create datasets
    # Check if we should use development subset for training
    use_dev_subset = (development_subset_path is not None and 
                     os.path.exists(development_subset_path))
    
    if use_dev_subset:
        # Load training data from development subset Parquet file
        print(f"Loading training data from development subset: {development_subset_path}")
        train_dataset = ParquetDataset(
            parquet_path=development_subset_path,
            feature_cols=feature_cols,
            label_to_int=new_label_to_int if scaler_save_dir is not None else label_mapping_orig['multiclass']['label_to_int']
        )
    else:
        # Use memmap-based training dataset (original approach)
        train_dataset = MemmapDataset(train_features_path, train_labels_path, num_training_samples, len(feature_cols),
                                      unknown_class_index=unknown_class_index, label_mapping=new_label_to_int if scaler_save_dir is not None else None)
    
    # Validation and test datasets always use memmap approach
    val_dataset = MemmapDataset(val_features_path, val_labels_path, num_val_samples, len(feature_cols),
                                unknown_class_index=unknown_class_index, label_mapping=new_label_to_int if scaler_save_dir is not None else None)
    test_dataset = MemmapDataset(test_features_path, test_labels_path, num_test_samples, len(feature_cols),
                                 unknown_class_index=unknown_class_index, label_mapping=new_label_to_int if scaler_save_dir is not None else None)
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, num_classes, len(feature_cols)


if __name__ == '__main__':
    # For testing purposes
    train_loader, val_loader, test_loader, num_classes, input_dim = create_data_loaders(
        data_dir='CICIOT23',
        batch_size=256,
        artifact_dir='results/baselines',
        development_subset_path=None,
        open_set=True,
        scaler_save_dir='experiments/results/test'
    )
    print(f"Number of classes: {num_classes}")
    print(f"Input dimension: {input_dim}")