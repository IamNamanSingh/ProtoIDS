"""
Dataset creation utilities for ProtoIDS.
Handles loading, preprocessing, and creating data loaders for CICIoT2023 and Edge-IIoTset.
"""

import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Optional, List, Tuple, Dict, Any
import warnings


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


def _create_data_loaders_ciciot2023(data_dir: str, batch_size: int = 256,
                                   artifact_dir: str = 'results/baselines',
                                   development_subset_path: Optional[str] = None,
                                   withheld_classes: Optional[list] = None,
                                   open_set: bool = False,
                                   scaler_save_dir: Optional[str] = None):
    """
    Original CICIoT2023 data loader logic (unchanged).
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


def _create_data_loaders_edgeiiot(data_dir: str, batch_size: int = 256,
                                  artifact_dir: str = 'results/baselines',
                                  development_subset_path: Optional[str] = None,
                                  withheld_classes: Optional[List[str]] = None,
                                  open_set: bool = False,
                                  scaler_save_dir: Optional[str] = None,
                                  dataset_name: str = 'edgeiiot'):
    """
    Edge-IIoTset data loader.
    Expects a single CSV file at <data_dir>/<dataset_name>.csv (or <data_dir>/NN-EdgeIIoT-dataset.csv/DNN-EdgeIIoT-dataset.csv?).
    We'll assume data_dir points to the directory containing the CSV file.
    The CSV file is expected to be named 'DNN-EdgeIIoT-dataset.csv' inside a subdirectory with same name?
    Actually from audit: the dataset is at '/c/Users/LENOVO/Desktop/workspace/Capstone Project/DNN-EdgeIIoT-dataset.csv/DNN-EdgeIIoT-dataset.csv'
    So data_dir could be the parent directory of the CSV file? We'll let the caller pass the full path to the CSV via data_dir?
    To keep consistent with CICIoT2023 where data_dir is the base directory containing train/validation/test subdirs,
    we will add a new argument maybe? But we cannot change the signature of create_data_loaders as used by train_protoids.py without modifying train_protoids.py.
    Instead we will interpret data_dir as the directory containing the CSV file, and we will look for a file named 'DNN-EdgeIIoT-dataset.csv' inside it.
    However the audit shows the CSV is inside a subdirectory with same name. We'll check for both possibilities.
    Simpler: we will expect the user to set --data_dir to the directory containing the CSV file (the parent of the CSV).
    We'll then construct the CSV path as os.path.join(data_dir, 'DNN-EdgeIIoT-dataset.csv', 'DNN-EdgeIIoT-dataset.csv')
    and if not found, try os.path.join(data_dir, 'DNN-EdgeIIoT-dataset.csv').
    We'll also support a direct path if they set data_dir to the CSV itself.
    We'll implement a helper to locate the CSV.
    """
    # Locate the CSV file
    possible_paths = [
        os.path.join(data_dir, 'DNN-EdgeIIoT-dataset.csv', 'DNN-EdgeIIoT-dataset.csv'),  # nested
        os.path.join(data_dir, 'DNN-EdgeIIoT-dataset.csv'),  # flat
        data_dir  # if data_dir itself is the CSV
    ]
    csv_path = None
    for p in possible_paths:
        if os.path.isfile(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"Could not find Edge-IIoTset CSV file in {data_dir}. Tried: {possible_paths}")

    print(f"Loading Edge-IIoTset dataset from: {csv_path}")

    # Load CSV, coerce all columns to numeric, fill NaN with 0
    # We'll also capture original dtypes to warn about coercion
    df_raw = pd.read_csv(csv_path, low_memory=False)
    original_dtypes = df_raw.dtypes.to_dict()

    # Coerce to numeric (excluding label columns)
    df = df_raw.copy()
    label_cols = ['Attack_label', 'Attack_type']
    cols_to_coerce = [col for col in df.columns if col not in label_cols]
    for col in cols_to_coerce:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify columns that had non-numeric values (produced NaN after coercion)
    coerced_cols = []
    for col in cols_to_coerce:
        if df[col].isna().any():
            coerced_cols.append(col)
    if coerced_cols:
        warnings.warn(f"Columns required coercion from non-numeric to numeric (NaN replaced with 0): {coerced_cols}")

    # Fill NaN with 0
    df = df.fillna(0)
    # Note: label columns remain as original strings

    # Define columns to remove (leakage, labels, etc.) as per audit
    label_cols = ['Attack_label', 'Attack_type']
    high_risk_cols = [
        'frame.time',
        'ip.src_host',
        'ip.dst_host',
        'arp.dst.proto_ipv4',
        'arp.src.proto_ipv4',
        'tcp.payload',
        'tcp.options',
        'mqtt.msg',
        'http.request.full_uri'
    ]
    additional_cols = [
        'mqtt.conack.flags',
        'mqtt.protoname',
        'mqtt.topic',
        'http.referer'
    ]
    cols_to_remove = label_cols + high_risk_cols + additional_cols

    # Remove label columns and leakage columns from features
    feature_cols = [col for col in df.columns if col not in cols_to_remove]
    print(f"Number of features after removing leakage and label columns: {len(feature_cols)}")

    # Prepare label mapping for Attack_type
    # We'll get unique Attack_type values from the original raw data (as strings)
    attack_type_series = df_raw['Attack_type']
    unique_attack_types = sorted(attack_type_series.unique())
    print(f"Unique Attack_type values: {unique_attack_types}")

    # Determine which classes are withheld
    if open_set and withheld_classes is not None and len(withheld_classes) > 0:
        # Use the provided withheld class names (strings)
        withheld_set = set(withheld_classes)
        # Validate that all withheld classes exist in the dataset
        unknown_in_data = withheld_set - set(unique_attack_types)
        if unknown_in_data:
            raise ValueError(f"The following withheld classes are not present in the dataset: {unknown_in_data}")
        known_set = set(unique_attack_types) - withheld_set
    else:
        # Default behavior: if open_set is True but no custom classes supplied, we fall back to CICIoT23 hardcoded list?
        # But for Edge-IIoT we should not hardcode CICIoT23 classes. We'll treat as no withholding (closed set) unless custom classes given.
        # We'll keep open_set=False effectively.
        withheld_set = set()
        known_set = set(unique_attack_types)
        open_set = False  # override to closed set if no withheld classes given

    # Map Attack_type to integer labels: known classes 0..len(known_set)-1, withheld to len(known_set)
    known_list = sorted(list(known_set))
    label_to_int = {cls: idx for idx, cls in enumerate(known_list)}
    unknown_class_index = len(known_list)  # if open_set else -1
    if not open_set:
        unknown_class_index = None  # not used

    # Function to map a raw Attack_type string to integer label
    def map_attack_type(val):
        if val in label_to_int:
            return label_to_int[val]
        else:
            # Withheld class
            return unknown_class_index

    # Apply mapping to create integer label column
    df['label_int'] = df['Attack_type'].apply(map_attack_type)

    # Now we need to split the data into train, val, test stratified by Attack_type (original string)
    # We'll do two splits: first split off test (15%), then split the remaining into train (70%) and val (15%).
    # Alternatively, we can split off val (15%) first, then split remaining into train (70%) and test (15%).
    # We'll use StratifiedShuffleSplit with n_splits=1, test_size=0.15, random_state=42 to get train+val vs test.
    # Then on the train+val portion, we split again with test_size=0.15/0.85 (so that val is 0.15 of original).

    X = df[feature_cols].values  # features
    y_str = df['Attack_type'].values  # stratify by original string labels

    # First split: train+val (70%) vs test (30%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_val_idx, test_idx in sss1.split(X, y_str):
        pass

    X_train_val = X[train_val_idx]
    y_train_val_str = y_str[train_val_idx]
    X_test = X[test_idx]
    y_test_str = y_str[test_idx]

    # Second split: split train+val into train (70% of original) and val (15% of original)
    # Relative sizes: we want train to be 0.70 of original, val 0.15 of original.
    # Given train_val is 0.70 of original, we need to take from train_val:
    # train fraction = 0.70 / 0.70 = 1.0? Wait we need to split train_val into train and val such that:
    # train size = 0.70 original, val size = 0.15 original.
    # Since train_val size = 0.70 original, we need to take train = (0.70/0.70) = 100% of train_val? That would leave 0 for val.
    # Actually we need to compute: from train_val (size 0.70), we want to leave out val of size 0.15 original, which is 0.15/0.70 = 0.2142857 of train_val.
    # So we split train_val with test_size = 0.2142857 (so that test becomes val) and train remains the rest.
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
    for train_idx, val_idx in sss2.split(X_train_val, y_train_val_str):
        pass

    X_train = X_train_val[train_idx]
    y_train_str = y_train_val_str[train_idx]
    X_val = X_train_val[val_idx]
    y_val_str = y_train_val_str[val_idx]

    # Now we have indices relative to the original dataframe:
    # train_idx (within train_val), val_idx (within train_val), test_idx (original)
    # Let's compute actual indices in original df:
    train_indices = train_val_idx[train_idx]
    val_indices = train_val_idx[val_idx]
    test_indices = test_idx

    # For open set, we need to ensure that withheld classes are excluded from training (but kept in val/test and mapped to unknown)
    # We already have label_int where withheld classes are mapped to unknown_class_index.
    # However we must also ensure that the scaler is fitted only on known training samples (i.e., those with label_int != unknown_class_index)
    # So we will compute known-only mask for training.

    # Determine constant columns from training data only (after removing leakage columns, but before scaling)
    # We'll use the training subset (original indices) and the feature columns.
    # We'll consider only known training samples (label_int != unknown_class_index) if open_set, else all training samples.
    if open_set:
        known_train_mask = df.iloc[train_indices]['label_int'] != unknown_class_index
        known_train_indices = train_indices[known_train_mask.values]
    else:
        known_train_indices = train_indices  # all training samples are known

    # Compute constant columns on known training data
    # We'll look at the feature columns subset
    train_known_features = df.iloc[known_train_indices][feature_cols]
    constant_cols = []
    for col in feature_cols:
        # Check if column has only one unique value (excluding NaN? but we have no NaN)
        if train_known_features[col].nunique() == 1:
            constant_cols.append(col)
    if constant_cols:
        print(f"Constant columns found in training data (will be removed): {constant_cols}")
        # Remove constant columns from feature columns
        feature_cols = [col for col in feature_cols if col not in constant_cols]
        # Update X arrays accordingly
        # We'll need to re-slice the data with the new feature columns
        X = df[feature_cols].values
        # Resplit indices? Instead we can recompute X_train, X_val, X_test using the new feature columns.
        # Let's do that:
        X_train = df.iloc[train_indices][feature_cols].values
        X_val = df.iloc[val_indices][feature_cols].values
        X_test = df.iloc[test_indices][feature_cols].values
        y_train_str = df.iloc[train_indices]['Attack_type'].values
        y_val_str = df.iloc[val_indices]['Attack_type'].values
        y_test_str = df.iloc[test_indices]['Attack_type'].values
        # Also update label_int series? We'll recompute later.

    # After removing constant columns, we need to recompute the label_int mapping? No, label mapping unchanged.
    # Now we have final feature columns.

    # Compute total raw training partition size
    total_raw_training = len(train_indices)
    known_training_for_scaler = len(known_train_indices)
    withheld_excluded = total_raw_training - known_training_for_scaler
    print(f"Total raw training partition (before withholding exclusion): {total_raw_training}")
    print(f"Known training samples used for scaler.fit: {known_training_for_scaler}")
    print(f"Withheld training samples excluded from scaler.fit: {withheld_excluded}")

    # Features matrix for all samples (after constant column removal)
    X_all = X  # X already holds df[feature_cols].values with final feature_cols

    # Split data: training uses only known samples; validation and test include all samples
    X_train = X_all[known_train_indices]
    y_train = df.iloc[known_train_indices]['label_int'].values.astype(np.int64)

    X_val = X_all[val_indices]
    y_val = df.iloc[val_indices]['label_int'].values.astype(np.int64)

    X_test = X_all[test_indices]
    y_test = df.iloc[test_indices]['label_int'].values.astype(np.int64)

    # Fit StandardScaler on known training samples only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Note: y_train, y_val, y_test already contain label_int values (with UNKNOWN index for withheld classes in val/test)

    # Determine number of classes
    if open_set:
        num_classes = len(known_list) + 1  # known + unknown
    else:
        num_classes = len(known_list)  # all classes known

    input_dim = len(feature_cols)

    # Save artifacts to experiment directories
    if scaler_save_dir is not None:
        print(f"Saving scaler and label mappings to {scaler_save_dir}")
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_save_dir, f'{dataset_name}_preprocessor.joblib'))
        # Create feature manifest
        manifest = {
            'original_columns': df_raw.columns.tolist(),
            'removed_columns': cols_to_remove + constant_cols,
            'retained_columns': feature_cols,
            'removal_reasons': {
                **{col: 'Label column' for col in label_cols},
                **{col: 'High leakage risk (IP, timestamp, port, payload)' for col in high_risk_cols},
                **{col: 'Recommended for removal in forensic report (environment-specific or leakage)' for col in additional_cols},
                **{col: 'Constant column (only one unique value)' for col in constant_cols}
            }
        }
        manifest_path = os.path.join(scaler_save_dir, f'{dataset_name}_feature_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        # Create label mapping (known classes only) in the same format as CICIoT2023
        label_mapping = {"multiclass": {"label_to_int": label_to_int}}
        label_mapping_path = os.path.join(scaler_save_dir, f'{dataset_name}_label_mapping.json')
        with open(label_mapping_path, 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Saved scaler, feature manifest, and label mapping to {scaler_save_dir}")

    # We will create memmap arrays for each split and then use MemmapDataset.
    # However we already have the scaled arrays in memory; we could directly create TensorDataset but to keep
    # consistency with the existing code path (which expects memmap files), we will write memmap files.
    # We'll create a processed base directory similar to CICIoT2023.
    if scaler_save_dir is not None:
        processed_base_dir = scaler_save_dir.replace('results', 'processed')
    else:
        processed_base_dir = os.path.join('experiments', 'processed', 'tmp')
    os.makedirs(processed_base_dir, exist_ok=True)
    print(f"Processed data will be saved to: {processed_base_dir}")

    # Training memmap
    processed_train_dir = os.path.join(processed_base_dir, 'train')
    os.makedirs(processed_train_dir, exist_ok=True)
    train_features_path = os.path.join(processed_train_dir, 'features.dat')
    train_labels_path = os.path.join(processed_train_dir, 'labels.dat')

    # Remove existing files if any
    try:
        if os.path.exists(train_features_path):
            os.remove(train_features_path)
    except PermissionError:
        with open(train_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(train_labels_path):
            os.remove(train_labels_path)
    except PermissionError:
        with open(train_labels_path, 'r+b') as f:
            f.truncate(0)

    train_features = np.memmap(train_features_path, dtype='float32', mode='w+', shape=(len(X_train_scaled), input_dim))
    train_labels = np.memmap(train_labels_path, dtype='int64', mode='w+', shape=(len(y_train),))
    train_features[:] = X_train_scaled
    train_labels[:] = y_train
    train_features.flush()
    train_labels.flush()

    # Validation memmap
    processed_val_dir = os.path.join(processed_base_dir, 'validation')
    os.makedirs(processed_val_dir, exist_ok=True)
    val_features_path = os.path.join(processed_val_dir, 'features.dat')
    val_labels_path = os.path.join(processed_val_dir, 'labels.dat')

    try:
        if os.path.exists(val_features_path):
            os.remove(val_features_path)
    except PermissionError:
        with open(val_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(val_labels_path):
            os.remove(val_labels_path)
    except PermissionError:
        with open(val_labels_path, 'r+b') as f:
            f.truncate(0)

    val_features = np.memmap(val_features_path, dtype='float32', mode='w+', shape=(len(X_val_scaled), input_dim))
    val_labels = np.memmap(val_labels_path, dtype='int64', mode='w+', shape=(len(y_val),))
    val_features[:] = X_val_scaled
    val_labels[:] = y_val
    val_features.flush()
    val_labels.flush()

    # Test memmap
    processed_test_dir = os.path.join(processed_base_dir, 'test')
    os.makedirs(processed_test_dir, exist_ok=True)
    test_features_path = os.path.join(processed_test_dir, 'features.dat')
    test_labels_path = os.path.join(processed_test_dir, 'labels.dat')

    try:
        if os.path.exists(test_features_path):
            os.remove(test_features_path)
    except PermissionError:
        with open(test_features_path, 'r+b') as f:
            f.truncate(0)
    try:
        if os.path.exists(test_labels_path):
            os.remove(test_labels_path)
    except PermissionError:
        with open(test_labels_path, 'r+b') as f:
            f.truncate(0)

    test_features = np.memmap(test_features_path, dtype='float32', mode='w+', shape=(len(X_test_scaled), input_dim))
    test_labels = np.memmap(test_labels_path, dtype='int64', mode='w+', shape=(len(y_test),))
    test_features[:] = X_test_scaled
    test_labels[:] = y_test
    test_features.flush()
    test_labels.flush()

    # Create datasets
    train_dataset = MemmapDataset(train_features_path, train_labels_path, len(y_train), input_dim,
                                  unknown_class_index=unknown_class_index if open_set else None,
                                  label_mapping=label_to_int if scaler_save_dir is not None else None)
    val_dataset = MemmapDataset(val_features_path, val_labels_path, len(y_val), input_dim,
                                unknown_class_index=unknown_class_index if open_set else None,
                                label_mapping=label_to_int if scaler_save_dir is not None else None)
    test_dataset = MemmapDataset(test_features_path, test_labels_path, len(y_test), input_dim,
                                 unknown_class_index=unknown_class_index if open_set else None,
                                 label_mapping=label_to_int if scaler_save_dir is not None else None)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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

    return train_loader, val_loader, test_loader, num_classes, input_dim


def create_data_loaders(data_dir: str, batch_size: int = 256,
                        artifact_dir: str = 'results/baselines',
                        development_subset_path: Optional[str] = None,
                        withheld_classes: Optional[list] = None,
                        open_set: bool = False,
                        scaler_save_dir: Optional[str] = None,
                        dataset_type: str = 'ciciot2023'):
    """
    Create data loaders for training, validation, and test sets.
    Handles both CICIoT2023 and Edge-IIoTset datasets.

    Args:
        data_dir: Path to dataset directory (for CICIoT2023: base dir containing train/validation/test;
                  for Edge-IIoTset: directory containing the CSV file or the CSV itself).
        batch_size: Batch size for data loaders
        artifact_dir: Directory containing feature and label mappings (for CICIoT2023)
        development_subset_path: Path to Parquet file for development subset (CICIoT2023 only)
        withheld_classes: List of class names (strings) to withhold from training (for open-set).
                          If None and dataset_type is 'ciciot2023' and open_set=True, uses hardcoded CICIoT23 class names.
                          If None and dataset_type is 'edgeiiot', treated as no withholding (closed set) unless open_set True and withheld_classes provided.
        open_set: Whether to perform open-set processing (withhold classes before scaling)
        scaler_save_dir: Directory to save scaler and label mappings (if open_set)
        dataset_type: Either 'ciciot2023' or 'edgeiiot' (default: 'ciciot2023')

    Returns:
        train_loader, val_loader, test_loader, num_classes, input_dim
    """
    if dataset_type == 'ciciot2023':
        # For backward compatibility, we need to map withheld_classes (if provided as strings) to indices.
        # The original function expects withheld_classes as list of indices.
        # We'll convert if needed.
        if withheld_classes is not None and isinstance(withheld_classes[0], str):
            # Convert string class names to indices using the CICIoT23 label mapping
            with open(os.path.join(artifact_dir, 'ciciot_label_mapping.json'), 'r') as f:
                label_mapping_orig = json.load(f)
            label_to_int = label_mapping_orig['multiclass']['label_to_int']
            withheld_classes_idx = [label_to_int[cls] for cls in withheld_classes if cls in label_to_int]
            # If any class not found, we will let the original function handle it (it will raise KeyError)
            return _create_data_loaders_ciciot2023(
                data_dir=data_dir,
                batch_size=batch_size,
                artifact_dir=artifact_dir,
                development_subset_path=development_subset_path,
                withheld_classes=withheld_classes_idx,
                open_set=open_set,
                scaler_save_dir=scaler_save_dir
            )
        else:
            # Assume already indices or None
            return _create_data_loaders_ciciot2023(
                data_dir=data_dir,
                batch_size=batch_size,
                artifact_dir=artifact_dir,
                development_subset_path=development_subset_path,
                withheld_classes=withheld_classes,
                open_set=open_set,
                scaler_save_dir=scaler_save_dir
            )
    elif dataset_type == 'edgeiiot':
        return _create_data_loaders_edgeiiot(
            data_dir=data_dir,
            batch_size=batch_size,
            artifact_dir=artifact_dir,
            development_subset_path=development_subset_path,
            withheld_classes=withheld_classes,
            open_set=open_set,
            scaler_save_dir=scaler_save_dir,
            dataset_name='edgeiiot'
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Supported: 'ciciot2023', 'edgeiiot'.")


if __name__ == '__main__':
    # For testing purposes
    print("Testing CICIoT2023 loader...")
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

    print("\nTesting Edge-IIoTset loader (will fail if CSV not found)...")
    try:
        train_loader, val_loader, test_loader, num_classes, input_dim = create_data_loaders(
            data_dir='/c/Users/LENOVO/Desktop/workspace/Capstone Project',  # adjust as needed
            batch_size=256,
            artifact_dir='results/baselines',
            open_set=True,
            withheld_classes=['MITM', 'Password', 'Vulnerability_scanner'],
            scaler_save_dir='experiments/results/edgeiiot_test',
            dataset_type='edgeiiot'
        )
        print(f"Number of classes: {num_classes}")
        print(f"Input dimension: {input_dim}")
    except Exception as e:
        print(f"Edge-IIoTset test failed (expected if CSV not accessible): {e}")