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