"""
Dataset creation utilities for ProtoIDS.
Handles loading, preprocessing, and creating data loaders for CICIoT2023 and Edge-IIoTset.
"""

import os
import re
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

    # First, read the header to get column names
    header_df = pd.read_csv(csv_path, nrows=0)
    all_columns = header_df.columns.tolist()

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

    # Determine initial feature columns (excluding leakage and label columns)
    initial_feature_cols = [col for col in all_columns if col not in cols_to_remove]
    print(f"Number of features after removing leakage and label columns: {len(initial_feature_cols)}")

    # Pass 1: Get unique Attack_type strings
    print("Pass 1: Collecting unique Attack_type values...")
    attack_type_set = set()
    chunksize = 100_000  # Adjust based on memory availability
    for chunk in pd.read_csv(csv_path, usecols=['Attack_type'], chunksize=chunksize):
        attack_type_set.update(chunk['Attack_type'].unique())

    print(f"Found {len(attack_type_set)} unique Attack_type values.")

    # Determine withheld classes (if provided) and validate
    if withheld_classes is not None and len(withheld_classes) > 0:
        withheld_set = set(withheld_classes)
        # Validate that all withheld classes exist in the dataset
        unknown_in_data = withheld_set - attack_type_set
        if unknown_in_data:
            raise ValueError(f"The following withheld classes are not present in the dataset: {unknown_in_data}")
    else:
        withheld_set = set()
        open_set = False  # override to closed set if no withheld classes given

    known_set = attack_type_set - withheld_set
    print(f"Known classes: {len(known_set)}, Withheld classes: {len(withheld_set)}")

    # Create mapping from Attack_type string to integer ID (unique per string)
    sorted_attack_types = sorted(list(attack_type_set))
    string_to_id = {string: idx for idx, string in enumerate(sorted_attack_types)}
    id_to_string = {idx: string for string, idx in string_to_id.items()}

    # Create label mapping: known classes -> 0..N-1, withheld -> unknown_class_index
    known_list = sorted(list(known_set))
    label_to_int = {cls: idx for idx, cls in enumerate(known_list)}
    unknown_class_index = len(known_list)  # if open_set else -1
    if not open_set:
        unknown_class_index = None  # not used

    # Pass 2: Read CSV to get attack_type_id for each row and store in array
    print("Pass 2: Reading Attack_type IDs for each row...")
    num_rows = 0
    attack_type_ids = []  # we'll store as list of ints, then convert to numpy array
    for chunk in pd.read_csv(csv_path, usecols=['Attack_type'], chunksize=chunksize):
        # Map each Attack_type string to its integer ID
        ids = chunk['Attack_type'].map(string_to_id).values
        attack_type_ids.extend(ids)
        num_rows += len(chunk)

    attack_type_ids = np.array(attack_type_ids, dtype=np.int32)
    print(f"Total rows: {num_rows}")

    # Now compute the label_int array (0..N-1 for known, unknown_class_index for withheld)
    label_int_array = np.full(num_rows, fill_value=unknown_class_index, dtype=np.int64)
    for string, label in label_to_int.items():
        sid = string_to_id[string]
        mask = (attack_type_ids == sid)
        label_int_array[mask] = label
    # Note: withheld classes remain as unknown_class_index (already set)

    # Now perform stratified split using attack_type_ids as the stratification variable (since it's unique per string)
    # We need a dummy X array for StratifiedShuffleSplit; we can use a column of zeros.
    X_dummy = np.zeros((num_rows, 1), dtype=np.float32)
    y_strat = attack_type_ids  # stratify by the integer ID (which maps 1:1 to Attack_type string)

    # First split: train+val (70%) vs test (15%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    for train_val_idx, test_idx in sss1.split(X_dummy, y_strat):
        pass

    # Second split: split train+val into train (70% of original) and val (15% of original)
    X_train_val = X_dummy[train_val_idx]
    y_train_val_strat = y_strat[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
    for train_idx, val_idx in sss2.split(X_train_val, y_train_val_strat):
        pass

    # Compute actual indices in original dataframe
    train_indices = train_val_idx[train_idx]
    val_indices = train_val_idx[val_idx]
    test_indices = test_idx

    print(f"Train+val+test sizes: {len(train_indices)} {len(val_indices)} {len(test_indices)}")

    # Determine known training samples (exclude withheld classes from training)
    known_train_mask = label_int_array[train_indices] != unknown_class_index
    known_train_indices = train_indices[known_train_mask]
    known_train_count = len(known_train_indices)
    total_train_count = len(train_indices)
    print(f"Total raw training partition (before withholding exclusion): {total_train_count}")
    print(f"Known training samples used for scaler.fit: {known_train_count}")
    print(f"Withheld training samples excluded from scaler.fit: {total_train_count - known_train_count}")

    # Precompute arrays for fast lookup in later passes
    is_known_train = np.zeros(num_rows, dtype=bool)
    is_known_train[known_train_indices] = True
    split_of_row = np.full(num_rows, fill_value=-1, dtype=np.int8)  # 0=train,1=val,2=test
    split_of_row[train_indices] = 0
    split_of_row[val_indices] = 1
    split_of_row[test_indices] = 2

    # Pass 3: Compute scaler statistics and constant column detection using only known training samples
    print("Pass 3: Computing statistics for scaling and constant column detection...")
    # Initialize online statistics for each initial feature column
    n_features = len(initial_feature_cols)
    col_sums = np.zeros(n_features, dtype=np.float64)
    col_sums_sq = np.zeros(n_features, dtype=np.float64)
    col_mins = np.full(n_features, fill_value=np.inf, dtype=np.float64)
    col_maxs = np.full(n_features, fill_value=-np.inf, dtype=np.float64)
    col_counts = np.zeros(n_features, dtype=np.int64)

    # We'll read the CSV in chunks, but only the initial feature columns
    # We need to know the column indices for initial_feature_cols in the CSV
    # We can use usecols=initial_feature_cols
    chunk_count = 0
    for chunk in pd.read_csv(csv_path, usecols=initial_feature_cols, chunksize=chunksize):
        chunk = chunk.apply(pd.to_numeric, errors='coerce').fillna(0)
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks...")
        # For each row in the chunk, we need to know its global row index
        # We can keep a running offset
        start_idx = (chunk_count - 1) * chunksize
        end_idx = start_idx + len(chunk)
        # Get the mask of known training samples in this chunk
        mask = is_known_train[start_idx:end_idx]
        if not np.any(mask):
            continue
        # Extract the features for known training samples in this chunk and convert to numeric
        features = chunk.iloc[mask].apply(pd.to_numeric, errors='coerce').fillna(0).values  # shape (n_samples, n_features)
        # Update online statistics
        col_sums += np.sum(features, axis=0)
        col_sums_sq += np.sum(features ** 2, axis=0)
        col_mins = np.minimum(col_mins, np.min(features, axis=0))
        col_maxs = np.maximum(col_maxs, np.max(features, axis=0))
        col_counts += np.sum(mask)

    # Compute mean and std
    means = col_sums / col_counts
    # Population variance: (sum_sq - sum*sum/count) / count
    variances = (col_sums_sq - col_sums * col_sums / col_counts) / col_counts
    # Avoid negative variance due to floating point
    variances = np.maximum(variances, 0.0)
    stds = np.sqrt(variances)

    # Detect constant columns: where min == max (or std == 0)
    constant_mask = (col_mins == col_maxs)  # or stds == 0
    constant_cols = [initial_feature_cols[i] for i in range(n_features) if constant_mask[i]]
    if constant_cols:
        print(f"Constant columns found in training data (will be removed): {constant_cols}")
    # Final feature columns: exclude constant columns
    final_feature_cols = [initial_feature_cols[i] for i in range(n_features) if not constant_mask[i]]
    input_dim = len(final_feature_cols)
    print(f"Number of features after removing constant columns: {input_dim}")

    # Pass 4: Write memmap files
    print("Pass 4: Creating memmap arrays...")
    # Determine sizes for each split
    train_count = known_train_count
    val_count = len(val_indices)
    test_count = len(test_indices)

    processed_base_dir = os.path.join('experiments', 'processed', 'edgeiiot_open_set_k3_lambda01')
    if scaler_save_dir is not None:
        processed_base_dir = scaler_save_dir.replace('results', 'processed')
    os.makedirs(processed_base_dir, exist_ok=True)
    print(f"Processed data will be saved to: {processed_base_dir}")

    # Training memmap (only known training samples)
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

    train_features = np.memmap(train_features_path, dtype='float32', mode='w+', shape=(train_count, input_dim))
    train_labels = np.memmap(train_labels_path, dtype='int64', mode='w+', shape=(train_count,))

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

    val_features = np.memmap(val_features_path, dtype='float32', mode='w+', shape=(val_count, input_dim))
    val_labels = np.memmap(val_labels_path, dtype='int64', mode='w+', shape=(val_count,))

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

    test_features = np.memmap(test_features_path, dtype='float32', mode='w+', shape=(test_count, input_dim))
    test_labels = np.memmap(test_labels_path, dtype='int64', mode='w+', shape=(test_count,))

    # We'll write row by row, keeping write pointers for each split
    train_write_idx = 0
    val_write_idx = 0
    test_write_idx = 0

    print("  Writing memmap arrays (this may take a while)...")
    # Read the CSV again, but we need both the initial feature columns and the Attack_type column for labeling
    # We'll read the initial feature columns and the Attack_type column together.
    usecols_pass4 = initial_feature_cols + ['Attack_type']
    chunk_count = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols_pass4, chunksize=chunksize):
        chunk[initial_feature_cols] = chunk[initial_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"  Processed {chunk_count} chunks for writing...")
        start_idx = (chunk_count - 1) * chunksize
        end_idx = start_idx + len(chunk)

        # Get the split assignment for this chunk
        splits_chunk = split_of_row[start_idx:end_idx]
        # Get the label_int for this chunk (from our precomputed array)
        labels_chunk = label_int_array[start_idx:end_idx]
        # Get the initial feature values for this chunk and convert to numeric
        features_chunk = chunk[initial_feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values  # shape (n_chunk, n_initial_features)
        # The Attack_type column is left as string for mapping
        attack_type_chunk = chunk['Attack_type'].values  # shape (n_chunk,)

        # Remove constant columns from features_chunk
        if len(constant_cols) > 0:
            # Create a mask for columns to keep
            keep_mask = np.array([col not in constant_cols for col in initial_feature_cols])
            features_chunk = features_chunk[:, keep_mask]
        # Now features_chunk has shape (n_chunk, input_dim)

        # Apply scaling: (x - mean) / std
        # We need to use the means and stds for the final feature columns (in the same order)
        # Since we removed constant columns, we need to select the corresponding means and stds
        # We have means and stds for all initial_feature_cols; we'll use the same keep_mask
        means_final = means[keep_mask] if len(constant_cols) > 0 else means
        stds_final = stds[keep_mask] if len(constant_cols) > 0 else stds
        # Avoid division by zero
        stds_final = np.where(stds_final == 0, 1.0, stds_final)
        features_scaled = (features_chunk - means_final) / stds_final

        # Write to memmap files based on split
        for i in range(len(chunk)):
            split = splits_chunk[i]
            label = labels_chunk[i]
            feat_row = features_scaled[i]
            if split == 0:  # train
                if label == unknown_class_index:
                    # skip withheld training samples
                    pass
                else:
                    train_features[train_write_idx] = feat_row
                    train_labels[train_write_idx] = label
                    train_write_idx += 1
            elif split == 1:  # val
                val_features[val_write_idx] = feat_row
                val_labels[val_write_idx] = label
                val_write_idx += 1
            elif split == 2:  # test
                test_features[test_write_idx] = feat_row
                test_labels[test_write_idx] = label
                test_write_idx += 1
            # else: should not happen

    # Flush to ensure data is written
    train_features.flush()
    train_labels.flush()
    val_features.flush()
    val_labels.flush()
    test_features.flush()
    test_labels.flush()

    print(f"  Training samples written: {train_write_idx}")
    print(f"  Validation samples written: {val_write_idx}")
    print(f"  Test samples written: {test_write_idx}")
    scaler = StandardScaler()
    scaler.mean_ = means_final
    scaler.scale_ = stds_final
    scaler.var_ = stds_final ** 2
    scaler.n_features_in_ = input_dim
    scaler.n_samples_seen_ = known_train_count

    # Create datasets
    train_dataset = MemmapDataset(train_features_path, train_labels_path, train_write_idx, input_dim,
                                  unknown_class_index=unknown_class_index if open_set else None,
                                  label_mapping=label_to_int if scaler_save_dir is not None else None)
    val_dataset = MemmapDataset(val_features_path, val_labels_path, val_write_idx, input_dim,
                                unknown_class_index=unknown_class_index if open_set else None,
                                label_mapping=label_to_int if scaler_save_dir is not None else None)
    test_dataset = MemmapDataset(test_features_path, test_labels_path, test_write_idx, input_dim,
                                 unknown_class_index=unknown_class_index if open_set else None,
                                 label_mapping=label_to_int if scaler_save_dir is not None else None)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Determine number of classes
    if open_set:
        num_classes = len(known_list) + 1  # known + unknown
    else:
        num_classes = len(known_list)  # all classes known

    print(f"Number of classes: {num_classes}")
    print(f"Input dimension: {input_dim}")

    # Save artifacts to experiment directories
    if scaler_save_dir is not None:
        print(f"Saving scaler and label mappings to {scaler_save_dir}")
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_save_dir, f'{dataset_name}_preprocessor.joblib'))
        # Create feature manifest
        manifest = {
            'original_columns': all_columns,
            'removed_columns': cols_to_remove + constant_cols,
            'retained_columns': final_feature_cols,
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

    return train_loader, val_loader, test_loader, num_classes, input_dim


# ===========================================================================
# X-IIoTID support
# ===========================================================================
# X-IIoTID is a single large CSV. We deliberately DISCOVER the schema at runtime
# rather than hard-coding column names, because the published release mixes
# versions (the Kaggle CSV has ~42 network features + 3 label levels, while the
# paper's final release advertises 68 features across several views). Discovery
# keeps the pipeline honest and version-tolerant.
# ===========================================================================

# Label columns. Mirrors of X-IIoTID disagree on naming, so we look for BOTH
# the descriptive names (the 42-column Kaggle CSV) and the class1/class2/class3
# triple (the 68-column release described in the paper, which is what the
# authors' own archive contains):
#     class1 -> 19 most granular attack types
#     class2 -> 10 attack categories
#     class3 -> binary Normal/Attack
# Granularity is decided from the data, not from the numbering, so the code
# keeps working if a mirror renames or reorders the columns.
XIIOTID_LABEL_NAMES = ['Sub-Category', 'Attack Type', 'Attack Category', 'Label']
XIIOTID_LABEL_PATTERN = re.compile(r'^class\d+$', re.IGNORECASE)

# Leakage filter. Matching on raw substrings is not safe here: 'ip' occurs in
# "Scr_ip_bytes" and 'id' occurs in "Avg_ideal_time", so plain substring
# matching silently deletes real measured features. Instead we tokenise the
# column name (underscores, punctuation AND camelCase) and only drop a column
# when it carries an identifier token and no measurement token.
#   Scr_port       -> [scr, port]         drop  (identifier)
#   Scr_ip_bytes   -> [scr, ip, bytes]    keep  (byte counter, not an address)
#   Avg_ideal_time -> [avg, ideal, time]  keep  ('id' is inside "ideal")
#   Timestamp      -> [timestamp]         drop
XIIOTID_ID_TOKENS = {
    'ip', 'ips', 'mac', 'macs', 'port', 'ports',
    'date', 'datetime', 'timestamp', 'time_stamp', 'id', 'ids', 'index',
    'uuid', 'address', 'addresses', 'flowid',
}
# A measurement unit means the column records a quantity rather than an
# identifier, so it wins over an incidental identifier token.
XIIOTID_MEASURE_TOKENS = {
    'bytes', 'byte', 'pkts', 'pkt', 'packet', 'packets', 'count', 'rate',
    'ratio', 'length', 'size', 'time', 'duration', 'total', 'avg', 'std',
    'num', 'proc', 's', 'kbmem', 'ldavg', 'tps', 'rtps', 'wtps',
}


def _tokenize_column(name: str) -> List[str]:
    """Split a column name into lowercase tokens on punctuation and camelCase."""
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', str(name))
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', spaced)
    return [t.lower() for t in re.split(r'[^A-Za-z0-9]+', spaced) if t]


def looks_like_xiiotid_identifier(name: str) -> bool:
    """True when the column name denotes an identifier/temporal/port field."""
    tokens = _tokenize_column(name)
    if not tokens:
        return False
    if any(t in XIIOTID_MEASURE_TOKENS for t in tokens):
        return False
    return any(t in XIIOTID_ID_TOKENS for t in tokens)

# Columns that encode a third-party intrusion-detection verdict rather than
# observed behaviour. They are not train/test leakage, but they let the model
# read an existing classifier's answer instead of learning from traffic, which
# inflates every metric. Dropped by default; set EXCLUDE_IDS_ALERTS=False to
# keep them for an ablation.
XIIOTID_IDS_ALERT_COLUMNS = ['anomaly_alert', 'OSSEC_alert', 'OSSEC_alert_level']
XIIOTID_EXCLUDE_IDS_ALERTS = True


def find_xiiotid_label_columns(columns: List[str]) -> List[str]:
    """Return every column that looks like a label level, in discovery order."""
    norm = {c.strip().lower(): c for c in columns}
    found: List[str] = []
    for name in XIIOTID_LABEL_NAMES:
        if name.lower() in norm:
            found.append(norm[name.lower()])
    for c in columns:
        if c not in found and XIIOTID_LABEL_PATTERN.match(c.strip()):
            found.append(c)
    return found


def rank_xiiotid_label_columns(csv_path: str,
                               columns: List[str]) -> List[Tuple[str, int]]:
    """
    Order the label columns from most to least granular using their cardinality.

    Reading only the label columns is cheap, and deciding the target from the
    data avoids hard-coding "class1 is the granular one" for a dataset that is
    mirrored under several layouts.
    """
    candidates = find_xiiotid_label_columns(columns)
    if not candidates:
        raise ValueError(
            f"No label column found. Looked for {XIIOTID_LABEL_NAMES} or a "
            f"'class<N>' column. Dataset columns: {columns}"
        )
    counts = pd.read_csv(csv_path, usecols=candidates, low_memory=False)
    ranked = sorted(((c, int(counts[c].nunique(dropna=True))) for c in candidates),
                    key=lambda kv: -kv[1])
    return ranked


def discover_xiiotid_columns(columns: List[str], label_col: Optional[str] = None,
                             exclude_ids_alerts: Optional[bool] = None
                             ) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Decide the target label column and which columns to drop.

    ``label_col`` forces the target; otherwise the most granular label column
    is chosen by the caller via rank_xiiotid_label_columns (it needs the file).

    Returns (label_col, retained_features, removal_reasons).
    """
    if exclude_ids_alerts is None:
        exclude_ids_alerts = XIIOTID_EXCLUDE_IDS_ALERTS

    norm = {c.strip().lower(): c for c in columns}
    label_columns = find_xiiotid_label_columns(columns)

    if label_col is not None:
        if label_col not in columns:
            raise ValueError(
                f"Requested label column '{label_col}' is not in the dataset. "
                f"Available label columns: {label_columns}")
    else:
        if not label_columns:
            raise ValueError(
                f"Could not find a label column among {XIIOTID_LABEL_NAMES} or a "
                f"'class<N>' column. Got: {columns}")
        # Most specific descriptive name, else lowest-numbered class<N>.
        named = [c for c in XIIOTID_LABEL_NAMES if c.lower() in norm]
        if named:
            label_col = norm[named[0].lower()]
        else:
            classlike = [c for c in label_columns if XIIOTID_LABEL_PATTERN.match(c.strip())]
            label_col = sorted(classlike, key=lambda c: int(re.sub(r'\D', '', c)))[0]

    # Every other label level is metadata, not a feature.
    reasons: Dict[str, str] = {}
    for c in label_columns:
        if c != label_col:
            reasons[c] = 'Label column (other granularity level)'

    for c in columns:
        if c in reasons or c == label_col:
            continue
        low = c.strip().lower()
        if exclude_ids_alerts and c in XIIOTID_IDS_ALERT_COLUMNS:
            reasons[c] = 'Third-party IDS verdict (anomaly_alert/OSSEC) - not observed behaviour'
        elif low.startswith('unnamed'):
            reasons[c] = 'Unnamed index column'
        elif looks_like_xiiotid_identifier(c):
            reasons[c] = (f'Identifier/temporal/port column (leakage risk) '
                          f'- tokens={_tokenize_column(c)}')
    features = [c for c in columns if c not in reasons and c != label_col]
    return label_col, features, reasons


def suggest_xiiotid_withheld(data_dir: str, label_col: Optional[str] = None,
                             k: int = 3) -> Tuple[str, List[str], pd.Series]:
    """
    Inspect the X-IIoTID CSV and propose a default open-set withholding.

    Picks the k most frequent NON-dominant (attack) classes. 'Normal' (or any
    class containing 'normal'/'benign') is never withheld because removing normal
    traffic from training would make open-set evaluation meaningless.

    Returns (label_col, withheld_classes, class_counts Series).
    """
    df, label_col_found = _read_xiiotid_label(data_dir, label_col)
    counts = df[label_col_found].value_counts()
    attack_classes = [c for c in counts.index
                      if not any(kw in str(c).lower() for kw in ('normal', 'benign', 'clean'))]
    withheld = sorted(attack_classes, key=lambda c: -counts[c])[:k]
    return label_col_found, withheld, counts


def _locate_xiiotid_csv(data_dir: str) -> str:
    """Find the X-IIoTID CSV under data_dir (handles nested download dirs)."""
    if os.path.isfile(data_dir):
        return data_dir
    candidates = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found under {data_dir}")
    # Prefer names that mention the dataset, else the largest CSV.
    named = [c for c in candidates if 'x-iiotid' in os.path.basename(c).lower()
             or 'xiiotid' in os.path.basename(c).lower()]
    if named:
        return max(named, key=os.path.getsize)
    return max(candidates, key=os.path.getsize)


def _read_xiiotid_label(data_dir: str, label_col: Optional[str] = None,
                        nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """Read just the label column(s); used for discovery and class listing."""
    csv_path = _locate_xiiotid_csv(data_dir)
    header = pd.read_csv(csv_path, nrows=0)
    if label_col is None:
        # Most granular label level, decided from the data.
        ranked = rank_xiiotid_label_columns(csv_path, header.columns.tolist())
        label_col = ranked[0][0]
    df = pd.read_csv(csv_path, usecols=[label_col], nrows=nrows, low_memory=False)
    return df, label_col


def _create_data_loaders_xiiotid(data_dir: str, batch_size: int = 256,
                                 artifact_dir: str = 'results/baselines',
                                 development_subset_path: Optional[str] = None,
                                 withheld_classes: Optional[List[str]] = None,
                                 open_set: bool = False,
                                 scaler_save_dir: Optional[str] = None,
                                 dataset_name: str = 'xiiotid',
                                 label_col: Optional[str] = None,
                                 max_rows: Optional[int] = None,
                                 split_strategy: str = 'random',
                                 time_col: Optional[str] = None,
                                 train_frac: float = 0.70,
                                 val_frac: float = 0.15) -> Tuple:
    """
    X-IIoTset data loader with a true open-set protocol.

    Protocol (matches CICIoT2023 / Edge-IIoTset):
      1. Discover label column + drop leakage columns.
      2. Stratified 70/15/15 split over ALL rows (unknown classes present in
         val/test, absent from train).
      3. Fit scaler on KNOWN TRAIN rows only; val/test transformed with it.
      4. Withheld classes map to an explicit unknown index; never seen in train.

    max_rows: optional cap (for quick smoke tests on huge files).
    """
    csv_path = _locate_xiiotid_csv(data_dir)
    print(f"Loading X-IIoTID dataset from: {csv_path}")

    # ---- discover schema ----
    header = pd.read_csv(csv_path, nrows=0)
    all_columns = header.columns.tolist()
    ranked = rank_xiiotid_label_columns(csv_path, all_columns)
    if label_col is None:
        label_col = ranked[0][0]
    elif label_col not in all_columns:
        raise ValueError(f"Requested label column '{label_col}' not in dataset. "
                         f"Available label columns: {[c for c, _ in ranked]}")
    _, feature_cols, reasons = discover_xiiotid_columns(all_columns, label_col=label_col)

    print("Label levels found (most -> least granular):")
    for col, nun in ranked:
        mark = "  <-- target" if col == label_col else ""
        print(f"  {col:<16} {nun:>3} distinct{mark}")
    print(f"Target label column: {label_col}")
    print(f"Features before constant-column removal: {len(feature_cols)}")

    # ---- read the label column to enumerate classes ----
    label_series = pd.read_csv(csv_path, usecols=[label_col], low_memory=False)
    if max_rows is not None:
        label_series = label_series.iloc[:max_rows]
    y_raw = label_series[label_col].astype(str).values
    num_rows = len(y_raw)
    print(f"Total rows: {num_rows}")

    classes = sorted(np.unique(y_raw).tolist())
    print(f"Found {len(classes)} classes: {classes}")

    if withheld_classes is not None and len(withheld_classes) > 0:
        missing = set(withheld_classes) - set(classes)
        if missing:
            raise ValueError(f"Withheld classes not present in dataset: {missing}")
        withheld_set = set(withheld_classes)
    else:
        withheld_set = set()
        open_set = False  # no withholding => closed set

    known_list = [c for c in classes if c not in withheld_set]
    label_to_int = {c: i for i, c in enumerate(known_list)}
    unknown_class_index = len(known_list) if open_set else None
    print(f"Known classes: {len(known_list)}, Withheld: {len(withheld_set)}")
    if withheld_set:
        print(f"Withheld (unknown): {sorted(withheld_set)}")

    # Map every row to either a known int or the unknown int
    y_int = np.full(num_rows, fill_value=(unknown_class_index if open_set else -1), dtype=np.int64)
    for cls, idx in label_to_int.items():
        y_int[y_raw == cls] = idx
    if not open_set:
        y_int[y_int == -1] = 0  # shouldn't happen, but guard

    # ---- split ----
    # 'random'  stratified 70/15/15 (fast to iterate, but time-adjacent flows end
    #           up on both sides of the boundary, so it flatters the model)
    # 'temporal' contiguous time blocks, earliest -> train, latest -> test. This
    #           is the honest generalisation test for an IDS: the model must
    #           handle traffic it has never seen in time, which is what actually
    #           happens in deployment. X-IIoTID spans 303 days and the class mix
    #           drifts heavily over that window, so the two splits are not
    #           interchangeable.
    if split_strategy == 'random':
        X_dummy = np.zeros((num_rows, 1), dtype=np.float32)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
        for train_val_idx, test_idx in sss1.split(X_dummy, y_raw):
            pass
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac / (1.0 - val_frac), random_state=42)
        for train_rel, val_rel in sss2.split(X_dummy[train_val_idx],
                                             y_raw[train_val_idx]):
            pass
        train_indices = train_val_idx[train_rel]
        val_indices = train_val_idx[val_rel]
        test_indices = test_idx
    elif split_strategy == 'temporal':
        if time_col is None:
            time_col = 'Timestamp' if 'Timestamp' in all_columns else 'Date'
        if time_col not in all_columns:
            raise ValueError(
                f"split_strategy='temporal' needs a time column; {time_col!r} "
                f"not in {all_columns}. Pass --xiiotid_time_col.")
        tvals = pd.read_csv(csv_path, usecols=[time_col], nrows=max_rows,
                            low_memory=False)[time_col]
        if max_rows is not None:
            tvals = tvals.iloc[:max_rows]
        tnum = pd.to_numeric(tvals, errors='coerce')
        if tnum.notna().sum() < len(tnum) * 0.5:
            parsed = pd.to_datetime(tvals, errors='coerce', format='mixed')
            tnum = parsed.astype('int64', errors='coerce') / 1e9
        # X-IIoTID has a few hundred rows whose Timestamp is not numeric (the
        # column is inferred as mixed type, so some values arrive as the strings
        # "TRUE"/"FALSE"). They are imputed with the median timestamp rather than
        # dropped, because every downstream array is indexed by absolute CSV row
        # position and removing rows here would silently misalign them. The count
        # is reported rather than hidden.
        bad = tnum.isna()
        if bad.any():
            n_bad = int(bad.sum())
            print(f"  NOTE: {n_bad} rows ({100 * n_bad / len(tnum):.3f}%) have an "
                  f"unparseable {time_col!r}; imputed with the median timestamp")
            tnum = tnum.fillna(tnum.median())
        order = np.argsort(tnum.to_numpy(), kind='stable')
        n = len(order)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_indices = order[:n_train]
        val_indices = order[n_train:n_train + n_val]
        test_indices = order[n_train + n_val:]
        print(f"temporal split on {time_col!r}: "
              f"train {pd.to_datetime(tnum.iloc[train_indices].min(), unit='s').date()} .. "
              f"{pd.to_datetime(tnum.iloc[train_indices].max(), unit='s').date()}")
        print(f"  val   {pd.to_datetime(tnum.iloc[val_indices].min(), unit='s').date()} .. "
              f"{pd.to_datetime(tnum.iloc[val_indices].max(), unit='s').date()}")
        print(f"  test  {pd.to_datetime(tnum.iloc[test_indices].min(), unit='s').date()} .. "
              f"{pd.to_datetime(tnum.iloc[test_indices].max(), unit='s').date()}")
        for nm, idx in (('train', train_indices), ('val', val_indices),
                        ('test', test_indices)):
            seen = len(set(y_raw[idx]))
            unk = len(withheld_set & set(y_raw[idx]))
            print(f"  {nm:<6} {len(idx):>7} rows  {seen:>3} classes  "
                  f"(withheld present: {unk})")
    else:
        raise ValueError(f"split_strategy must be 'random' or 'temporal', "
                         f"got {split_strategy!r}")
    print(f"Split sizes (raw): train {len(train_indices)}, val {len(val_indices)}, test {len(test_indices)}")

    # Known-train mask: drop withheld rows from training entirely
    is_known_train = np.zeros(num_rows, dtype=bool)
    is_known_train[train_indices] = True
    if open_set:
        is_known_train &= (y_int != unknown_class_index)
    known_train_count = int(is_known_train.sum())
    print(f"Known training samples (after withholding): {known_train_count}")

    split_of_row = np.full(num_rows, -1, dtype=np.int8)
    split_of_row[train_indices] = 0
    split_of_row[val_indices] = 1
    split_of_row[test_indices] = 2

    # ---- Pass 1: column stats on known-train rows only (chunked) ----
    print("Pass 1: computing per-column stats on known-train rows...")
    n_features = len(feature_cols)
    col_sum = np.zeros(n_features, dtype=np.float64)
    col_sumsq = np.zeros(n_features, dtype=np.float64)
    col_min = np.full(n_features, np.inf, dtype=np.float64)
    col_max = np.full(n_features, -np.inf, dtype=np.float64)
    col_count = 0

    chunksize = 100_000
    offset = 0
    for chunk in pd.read_csv(csv_path, usecols=feature_cols, chunksize=chunksize,
                              nrows=max_rows, low_memory=False):
        Xc = chunk.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
        end = offset + len(chunk)
        mask = is_known_train[offset:end]
        if np.any(mask):
            Xk = Xc[mask]
            col_sum += Xk.sum(axis=0)
            col_sumsq += (Xk ** 2).sum(axis=0)
            col_min = np.minimum(col_min, Xk.min(axis=0))
            col_max = np.maximum(col_max, Xk.max(axis=0))
            col_count += int(mask.sum())
        offset = end

    means = col_sum / max(col_count, 1)
    var = np.maximum((col_sumsq - col_sum ** 2 / max(col_count, 1)) / max(col_count, 1), 0.0)
    stds = np.sqrt(var)
    constant_mask = (col_min == col_max)
    constant_cols = [feature_cols[i] for i in range(n_features) if constant_mask[i]]
    if constant_cols:
        print(f"Constant columns in known-train (dropped): {constant_cols}")
    keep_mask = ~constant_mask
    final_features = [feature_cols[i] for i in range(n_features) if keep_mask[i]]
    means_final = means[keep_mask]
    stds_final = np.where(stds[keep_mask] == 0, 1.0, stds[keep_mask])
    input_dim = len(final_features)
    print(f"Final input dimension: {input_dim}")

    # ---- Pass 2: write scaled memmaps (chunked, vectorized) ----
    print("Pass 2: scaling and writing memmap arrays...")
    processed_base = (scaler_save_dir.replace('results', 'processed')
                      if scaler_save_dir else os.path.join('experiments', 'processed', 'xiiotid'))
    os.makedirs(processed_base, exist_ok=True)

    def _open_memmap(split_name, n):
        d = os.path.join(processed_base, split_name)
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, 'features.dat')
        lp = os.path.join(d, 'labels.dat')
        for p in (fp, lp):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        feats = np.memmap(fp, dtype='float32', mode='w+', shape=(n, input_dim))
        labs = np.memmap(lp, dtype='int64', mode='w+', shape=(n,))
        return feats, labs

    # counts of what will actually be written per split
    n_train_write = known_train_count
    n_val_write = int((split_of_row == 1).sum())
    n_test_write = int((split_of_row == 2).sum())
    train_features, train_labels = _open_memmap('train', n_train_write)
    val_features, val_labels = _open_memmap('validation', n_val_write)
    test_features, test_labels = _open_memmap('test', n_test_write)

    w_train = w_val = w_test = 0
    offset = 0
    for chunk in pd.read_csv(csv_path, usecols=feature_cols, chunksize=chunksize,
                              nrows=max_rows, low_memory=False):
        Xc = chunk.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
        end = offset + len(chunk)
        splits = split_of_row[offset:end]
        y_slice = y_int[offset:end]
        Xc = (Xc[:, keep_mask] - means_final) / stds_final
        Xc = np.clip(Xc, -5, 5).astype(np.float32)

        # train: only known-train rows
        m = (splits == 0) & is_known_train[offset:end]
        if np.any(m):
            train_features[w_train:w_train + int(m.sum())] = Xc[m]
            train_labels[w_train:w_train + int(m.sum())] = y_slice[m]
            w_train += int(m.sum())

        m = splits == 1
        if np.any(m):
            val_features[w_val:w_val + int(m.sum())] = Xc[m]
            val_labels[w_val:w_val + int(m.sum())] = y_slice[m]
            w_val += int(m.sum())

        m = splits == 2
        if np.any(m):
            test_features[w_test:w_test + int(m.sum())] = Xc[m]
            test_labels[w_test:w_test + int(m.sum())] = y_slice[m]
            w_test += int(m.sum())

        offset = end

    train_features.flush(); train_labels.flush()
    val_features.flush(); val_labels.flush()
    test_features.flush(); test_labels.flush()
    print(f"Written: train {w_train}, val {w_val}, test {w_test}")

    # ---- rebuild a fitted scaler object for downstream use ----
    scaler = StandardScaler()
    scaler.mean_ = means_final
    scaler.scale_ = stds_final
    scaler.var_ = stds_final ** 2
    scaler.n_features_in_ = input_dim
    scaler.n_samples_seen_ = col_count

    train_dataset = MemmapDataset(*_split_paths(processed_base, 'train'),
                                  w_train, input_dim,
                                  unknown_class_index=unknown_class_index, label_mapping=label_to_int)
    val_dataset = MemmapDataset(*_split_paths(processed_base, 'validation'),
                                w_val, input_dim,
                                unknown_class_index=unknown_class_index, label_mapping=label_to_int)
    test_dataset = MemmapDataset(*_split_paths(processed_base, 'test'),
                                 w_test, input_dim,
                                 unknown_class_index=unknown_class_index, label_mapping=label_to_int)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    num_classes = (len(known_list) + 1) if open_set else len(known_list)

    # ---- split manifest: lets a sweep re-train without re-parsing the CSV ----
    # Re-reading 355 MB of CSV for every hyper-parameter trial is the dominant
    # cost of a sweep, and it is identical work every time because it depends
    # only on (data, withheld classes, seed). Recording the shape here allows
    # create_loaders_from_cache() to rebuild the loaders in milliseconds.
    split_meta = {
        'dataset': dataset_name,
        'processed_dir': processed_base,
        'csv_path': csv_path,
        'label_column': label_col,
        'input_dim': input_dim,
        'retained_features': final_features,
        'removed_columns': list(reasons.keys()) + constant_cols,
        'withheld_classes': sorted(withheld_set) if open_set else [],
        'known_classes': known_list,
        'label_to_int': {c: int(i) for c, i in label_to_int.items()},
        'unknown_class_index': unknown_class_index,
        'open_set': bool(open_set),
        'num_classes': num_classes,
        'split_strategy': split_strategy,
        'time_col': time_col if split_strategy == 'temporal' else None,
        'train_frac': train_frac if split_strategy == 'temporal' else None,
        'val_frac': val_frac if split_strategy == 'temporal' else None,
        'counts': {'train': w_train, 'validation': w_val, 'test': w_test},
        'scaler_n_samples_seen': int(col_count),
        'split_seed': 42,
    }
    with open(os.path.join(processed_base, 'split_meta.json'), 'w') as f:
        json.dump(split_meta, f, indent=2)
    print(f"Split manifest: {os.path.join(processed_base, 'split_meta.json')}")

    # ---- persist artifacts ----
    if scaler_save_dir is not None:
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_save_dir, f'{dataset_name}_preprocessor.joblib'))
        manifest = {
            'original_columns': all_columns,
            'removed_columns': list(reasons.keys()) + constant_cols,
            'retained_columns': final_features,
            'removal_reasons': {**reasons,
                                **{c: 'Constant column (single unique value in known-train)'
                                   for c in constant_cols}},
            'label_column': label_col,
        }
        with open(os.path.join(scaler_save_dir, f'{dataset_name}_feature_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        label_mapping = {
            "multiclass": {"label_to_int": {c: int(i) for c, i in label_to_int.items()}},
            "classes": known_list,
            "withheld_classes": sorted(withheld_set) if open_set else [],
            "unknown_class_index": unknown_class_index,
        }
        with open(os.path.join(scaler_save_dir, f'{dataset_name}_label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Saved scaler, manifest, and label mapping to {scaler_save_dir}")

    return train_loader, val_loader, test_loader, num_classes, input_dim


def _split_paths(base, split_name):
    """(features_path, labels_path) for a processed split subdirectory."""
    return (os.path.join(base, split_name, 'features.dat'),
            os.path.join(base, split_name, 'labels.dat'))


def create_loaders_from_cache(processed_dir: str, batch_size: int = 256,
                              shuffle_train: bool = True) -> Tuple:
    """
    Rebuild train/val/test loaders from a processed split on disk.

    Used by hyper-parameter sweeps: the expensive CSV parse and scaling depend
    only on the dataset and the withheld-class set, so they are done once and
    then reused for every trial. Returns the same tuple as create_data_loaders.
    """
    meta_path = os.path.join(processed_dir, 'split_meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run the loader once with "
            f"scaler_save_dir set so the split manifest is written.")
    with open(meta_path) as f:
        meta = json.load(f)

    input_dim = meta['input_dim']
    counts = meta['counts']
    unknown_idx = meta.get('unknown_class_index')
    label_to_int = meta.get('label_to_int')

    def _make(split, shuffle):
        feats, labs = _split_paths(processed_dir, split)
        if not os.path.isfile(feats):
            raise FileNotFoundError(f"missing {feats}")
        ds = MemmapDataset(feats, labs, counts[split], input_dim,
                           unknown_class_index=unknown_idx, label_mapping=label_to_int)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    loaders = (_make('train', shuffle_train), _make('validation', False),
               _make('test', False))
    return (*loaders, meta['num_classes'], input_dim)


def create_data_loaders(data_dir: str, batch_size: int = 256,
                        artifact_dir: str = 'results/baselines',
                        development_subset_path: Optional[str] = None,
                        withheld_classes: Optional[list] = None,
                        open_set: bool = False,
                        scaler_save_dir: Optional[str] = None,
                        dataset_type: str = 'ciciot2023',
                        label_col: Optional[str] = None,
                        max_rows: Optional[int] = None,
                        split_strategy: str = 'random',
                        time_col: Optional[str] = None,
                        train_frac: float = 0.70,
                        val_frac: float = 0.15):
    """
    Create data loaders for training, validation, and test sets.
    Handles CICIoT2023, Edge-IIoTset and X-IIoTID datasets.

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
        dataset_type: 'ciciot2023', 'edgeiiot' or 'xiiotid' (default: 'ciciot2023')
        label_col: (xiiotid only) force a specific target label column; auto-detected if None
        max_rows: (xiiotid only) cap the number of CSV rows read, for fast smoke tests
        split_strategy: (xiiotid only) 'random' stratified, or 'temporal' contiguous
                        time blocks (earliest->train, latest->test)
        time_col / train_frac / val_frac: (xiiotid only) temporal split controls

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
    elif dataset_type == 'xiiotid':
        return _create_data_loaders_xiiotid(
            data_dir=data_dir,
            batch_size=batch_size,
            artifact_dir=artifact_dir,
            development_subset_path=development_subset_path,
            withheld_classes=withheld_classes,
            open_set=open_set,
            scaler_save_dir=scaler_save_dir,
            dataset_name='xiiotid',
            label_col=label_col,
            max_rows=max_rows,
            split_strategy=split_strategy,
            time_col=time_col,
            train_frac=train_frac,
            val_frac=val_frac,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. "
                         "Supported: 'ciciot2023', 'edgeiiot', 'xiiotid'.")


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