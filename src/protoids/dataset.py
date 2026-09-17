"""
ProtoIDS Dataset Module
Handles loading and preprocessing of CICIoT2023 data for ProtoIDS.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import copy
import warnings
warnings.filterwarnings('ignore')


class IdentityScaler:
    """A scaler that does nothing, used as a placeholder when we want to fit our own scaler."""
    def transform(self, X):
        return X


class CICIoT2023ProtoIDSDataset(Dataset):
    """
    Dataset class for CICIoT2023 data with preprocessing for ProtoIDS.
    """

    def __init__(self, data_dir: str, split: str = 'train',
                  artifact_dir: str = 'results/baselines',
                  development_subset_path: Optional[str] = None,
                  withheld_classes: Optional[list] = None,
                  scaler: Optional[StandardScaler] = None,
                  manifest: Optional[dict] = None,
                  label_mapping: Optional[dict] = None):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the CSV splits (train/validation/test)
            split: Which split to load ('train', 'validation', or 'test')
            artifact_dir: Directory containing preprocessing artifacts (used if scaler/manifest/label_mapping not provided)
            development_subset_path: Path to development subset Parquet (for train split only)
            withheld_classes: Optional list of class indices to exclude (for true open-set retrain)
            scaler: Optional pre-fitted scaler to use for transformation
            manifest: Optional feature manifest to use
            label_mapping: Optional label mapping to use
        """
        self.data_dir = data_dir
        self.split = split
        self.artifact_dir = artifact_dir

        # Load preprocessing artifacts if not provided
        if scaler is None or manifest is None or label_mapping is None:
            _scaler, _manifest, _label_mapping = self._load_artifacts()
            if scaler is None:
                scaler = _scaler
            if manifest is None:
                manifest = _manifest
            if label_mapping is None:
                label_mapping = _label_mapping

        self.scaler = scaler
        self.manifest = manifest
        self.label_mapping = label_mapping

        # Get feature columns
        self.feature_cols = self.manifest['retained_columns']
        self.label_to_int = self.label_mapping['multiclass']['label_to_int']
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        # Load data
        self.X, self.y_multi, self.y_bin = self._load_data(
            development_subset_path if split == 'train' else None
        )

        # True open-set: withhold classes from training (HANDOVER §4)
        if withheld_classes is not None and len(withheld_classes) > 0:
            keep_mask = ~np.isin(self.y_multi, withheld_classes)
            filtered = keep_mask.sum()
            print(f"Withholding classes {withheld_classes}: {len(self.y_multi)-filtered} samples removed, {filtered} kept for split={self.split}")
            self.X = self.X[keep_mask]
            self.y_multi = self.y_multi[keep_mask]

        # Update label mapping to reflect only classes present in the data
        if withheld_classes is not None and len(withheld_classes) > 0:
            if not (scaler is not None and manifest is not None and label_mapping is not None):
                # Get unique labels in the data
                unique_labels = np.unique(self.y_multi)
                # Create new mapping from original label to new contiguous index
                new_label_to_int = {int(label): idx for idx, label in enumerate(sorted(unique_labels))}
                new_int_to_label = {idx: int(label) for label, idx in new_label_to_int.items()}
                self.label_to_int = new_label_to_int
                self.int_to_label = new_int_to_label
                # Update the label_mapping dict for consistency
                self.label_mapping['multiclass']['label_to_int'] = new_label_to_int
                # Remap y_multi from old labeling to new labeling using lookup array
                if len(self.y_multi) > 0:
                    max_label = int(self.y_multi.max())
                    lookup = np.full(max_label + 1, -1, dtype=np.int64)
                    for orig_idx, new_idx in new_label_to_int.items():
                        if orig_idx <= max_label:
                            lookup[orig_idx] = new_idx
                    self.y_multi = lookup[self.y_multi.astype(np.int64)]

        # Convert to tensors
        self.X_tensor = torch.from_numpy(self.X).float()
        self.y_multi_tensor = torch.from_numpy(self.y_multi).long()

    def _load_artifacts(self):
        """Load preprocessing artifacts (scaler, manifest, label mapping)."""
        scaler_path = os.path.join(self.artifact_dir, 'ciciot_preprocessor.joblib')
        manifest_path = os.path.join(self.artifact_dir, 'ciciot_feature_manifest.json')
        label_mapping_path = os.path.join(self.artifact_dir, 'ciciot_label_mapping.json')

        scaler = joblib.load(scaler_path)

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)

        return scaler, manifest, label_mapping

    def _load_data(self, development_subset_path: Optional[str]):
        """
        Load and preprocess data.

        For training split, if development_subset_path is provided, load from there.
        Otherwise, load from the CSV file in data_dir.
        For validation/test splits, always load from CSV files.
        """
        if self.split == 'train' and development_subset_path is not None:
            # Load development subset
            return self._load_development_subset(development_subset_path)
        else:
            # Load from CSV file
            return self._load_csv_split()

    def _load_development_subset(self, parquet_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load development subset from Parquet file.
        """
        # Load the Parquet file
        df = pd.read_parquet(parquet_path)

        # Extract features and labels
        X_df = df[self.feature_cols].copy()
        y_multi_df = df['label_multi'].copy()

        # Convert labels to integers
        y_multi_int = y_multi_df.map(self.label_to_int).values

        # Parquet is already scaled via ciciot_preprocessing.py StandardScaler
        # Do NOT re-apply scaler (double-scaling → 915k outliers), just clip
        X_scaled = np.clip(X_df.values, -5, 5).astype(np.float32)

        return X_scaled, y_multi_int, None  # Binary labels not needed for ProtoIDS

    def _load_csv_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from CSV file for the specified split.
        """
        csv_path = os.path.join(self.data_dir, self.split, f'{self.split}.csv')
        df = pd.read_csv(csv_path)

        # Extract features and labels
        X_df = df[self.feature_cols].copy()
        y_multi_df = df['label'].copy()

        # Convert labels to integers
        y_multi_int = y_multi_df.map(self.label_to_int).values

        # Scale features using the pre-fitted scaler (raw CSV → scaled)
        X_scaled = self.scaler.transform(X_df.values)
        X_scaled = np.clip(X_scaled, -5, 5).astype(np.float32)

        return X_scaled, y_multi_int, None  # Binary labels not needed for ProtoIDS

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            tuple: (features, label)
        """
        return self.X_tensor[idx], self.y_multi_tensor[idx]

    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return len(self.label_to_int)

    def get_input_dim(self) -> int:
        """Return the number of input features."""
        return len(self.feature_cols)

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balancing (inverse frequency).

        Returns:
            Tensor of class weights
        """
        # Count samples per class
        unique, counts = np.unique(self.y_multi, return_counts=True)
        class_counts = np.zeros(len(self.label_to_int))
        for cls, count in zip(unique, counts):
            class_counts[cls] = count

        # Compute inverse frequency weights
        total_samples = len(self.y_multi)
        class_weights = total_samples / (len(self.label_to_int) * class_counts)
        class_weights[class_counts == 0] = 0  # Avoid division by zero

        return torch.from_numpy(class_weights).float()


def create_data_loaders(data_dir: str, batch_size: int = 256,
                       artifact_dir: str = 'results/baselines',
                       development_subset_path: str = 'experiments/data/ciciot_dev.parquet',
                       val_batch_size: Optional[int] = None,
                       withheld_classes: Optional[list] = None,
                       open_set: bool = False,
                       scaler_save_dir: Optional[str] = None):
    """
    Create data loaders for training, validation, and test sets.

    Args:
        data_dir: Directory containing the CSV splits
        batch_size: Batch size for training
        artifact_dir: Directory containing preprocessing artifacts
        development_subset_path: Path to development subset Parquet
        val_batch_size: Batch size for validation/test (defaults to batch_size)
        withheld_classes: Optional list of class indices to exclude from training (for open-set)
        open_set: If True, fit a new scaler on training data (after withholding) and use it for all splits.
                  Do not use development subset.
        scaler_save_dir: If provided and open_set=True, save the fitted scaler and manifest to this directory.

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    print(f"DEBUG: create_data_loaders called with open_set={open_set}, scaler_save_dir={scaler_save_dir}")
    if val_batch_size is None:
        val_batch_size = batch_size

    # Convert paths to absolute
    if not os.path.isabs(artifact_dir):
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))
        project_root = os.path.dirname(src_dir)
        artifact_dir = os.path.join(project_root, artifact_dir)

        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir)
        if development_subset_path is not None and not os.path.isabs(development_subset_path):
            development_subset_path = os.path.join(project_root, development_subset_path)
    if scaler_save_dir is not None and not os.path.isabs(scaler_save_dir):
        scaler_save_dir = os.path.join(os.path.dirname(__file__), '..', '..', scaler_save_dir)
        scaler_save_dir = os.path.abspath(scaler_save_dir)

    if open_set:
        # For open-set: we will fit a scaler on training data (after withholding)
        # Load original manifest and label_mapping
        manifest_path = os.path.join(artifact_dir, 'ciciot_feature_manifest.json')
        label_mapping_path = os.path.join(artifact_dir, 'ciciot_label_mapping.json')
        with open(manifest_path, 'r') as f:
            manifest_orig = json.load(f)
        with open(label_mapping_path, 'r') as f:
            label_mapping_orig = json.load(f)

        # Load CSV files manually to have full control over processing
        train_csv_path = os.path.join(data_dir, 'train', 'train.csv')
        val_csv_path = os.path.join(data_dir, 'validation', 'validation.csv')
        test_csv_path = os.path.join(data_dir, 'test', 'test.csv')

        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)
        test_df = pd.read_csv(test_csv_path)

        # Extract feature columns
        feature_cols = manifest_orig['retained_columns']

        # Process training data
        # Extract features and labels
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df['label'].values  # string labels

        # Convert string labels to original indices using original label mapping
        # Use -1 for missing labels (will be treated as unknown)
        y_train_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_train_raw)

        # Apply withholding: remove samples with original class indices in withheld_classes
        # Also remove samples with invalid labels (mapped to -1)
        if withheld_classes is not None and len(withheld_classes) > 0:
            keep_mask = (~np.isin(y_train_orig, withheld_classes)) & (y_train_orig != -1)
            X_train_raw = X_train_raw[keep_mask]
            y_train_orig = y_train_orig[keep_mask]
            print(f"Withholding classes {withheld_classes}: {len(y_train_raw) - len(y_train_orig)} samples removed, {len(y_train_orig)} kept for split=train")

        # Fit scaler on training features
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_train_scaled = np.clip(X_train_scaled, -5, 5).astype(np.float32)

        # Compute new label mapping from training labels (after withholding)
        unique_labels = np.unique(y_train_orig)
        new_label_to_int = {int(label): idx for idx, label in enumerate(sorted(unique_labels))}
        new_int_to_label = {idx: int(label) for label, idx in new_label_to_int.items()}

        # Convert training labels from original indices to new indices
        y_train_new = np.vectorize(new_label_to_int.get)(y_train_orig)

        # Process validation data
        X_val_raw = val_df[feature_cols].values
        y_val_raw = val_df['label'].values
        # Map string labels to original indices, using -1 for missing labels
        y_val_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_val_raw)
        # No withholding for validation
        X_val_scaled = scaler.transform(X_val_raw)
        X_val_scaled = np.clip(X_val_scaled, -5, 5).astype(np.float32)
        # For validation:
        #   - If label is missing (-1) or in withheld_classes -> unknown class (label = num_known_classes)
        #   - If label is not in new_label_to_int -> unknown class (unseen in training)
        #   - Otherwise -> known class (get new label mapping)

        # Create lookup array for new_label_to_int for efficient mapping
        if new_label_to_int:
            max_key = max(new_label_to_int.keys())
            lookup = np.full(max_key + 1, -1, dtype=int)
            for k, v in new_label_to_int.items():
                lookup[k] = v
        else:
            lookup = np.array([], dtype=int)

        # Determine unknown labels: missing (-1), withheld, or not in new_label_to_int (unseen in training)
        is_unknown_val = (y_val_orig == -1) | np.isin(y_val_orig, withheld_classes) | (~np.isin(y_val_orig, list(new_label_to_int.keys())))
        # Default all to unknown class
        y_val_new = np.full_like(y_val_orig, len(new_label_to_int), dtype=int)
        # For known labels, map using lookup array
        known_mask = ~is_unknown_val
        if np.any(known_mask):
            y_val_new[known_mask] = lookup[y_val_orig[known_mask]]

        # Process test data
        X_test_raw = test_df[feature_cols].values
        y_test_raw = test_df['label'].values
        # Map string labels to original indices, using -1 for missing labels
        y_test_orig = np.vectorize(lambda x: label_mapping_orig['multiclass']['label_to_int'].get(x, -1))(y_test_raw)
        # No withholding for test
        X_test_scaled = scaler.transform(X_test_raw)
        X_test_scaled = np.clip(X_test_scaled, -5, 5).astype(np.float32)
        # For test:
        #   - If label is missing (-1) or in withheld_classes -> unknown class (label = num_known_classes)
        #   - If label is not in new_label_to_int -> unknown class (unseen in training)
        #   - Otherwise -> known class (get new label mapping)

        # Determine unknown labels: missing (-1), withheld, or not in new_label_to_int (unseen in training)
        is_unknown_test = (y_test_orig == -1) | np.isin(y_test_orig, withheld_classes) | (~np.isin(y_test_orig, list(new_label_to_int.keys())))
        # Default all to unknown class
        y_test_new = np.full_like(y_test_orig, len(new_label_to_int), dtype=int)
        # For known labels, map using lookup array
        known_mask_test = ~is_unknown_test
        if np.any(known_mask_test):
            y_test_new[known_mask_test] = lookup[y_test_orig[known_mask_test]]

        # Create TensorDatasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled).float(),
            torch.from_numpy(y_train_new).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_scaled).float(),
            torch.from_numpy(y_val_new).long()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test_scaled).float(),
            torch.from_numpy(y_test_new).long()
        )

        # Save artifacts if requested
        if scaler_save_dir is not None:
            print(f"DEBUG: About to create directory {scaler_save_dir}")
            os.makedirs(scaler_save_dir, exist_ok=True)
            print(f"DEBUG: Directory created")
            print(f"DEBUG: About to save scaler to {os.path.join(scaler_save_dir, 'ciciot_preprocessor.joblib')}")
            joblib.dump(scaler, os.path.join(scaler_save_dir, 'ciciot_preprocessor.joblib'))
            print(f"DEBUG: Scaler saved")
            print(f"DEBUG: About to save manifest to {os.path.join(scaler_save_dir, 'ciciot_feature_manifest.json')}")
            with open(os.path.join(scaler_save_dir, 'ciciot_feature_manifest.json'), 'w') as f:
                json.dump(manifest_orig, f, indent=2)
            print(f"DEBUG: Manifest saved")
            print(f"DEBUG: About to save label mapping to {os.path.join(scaler_save_dir, 'ciciot_label_mapping.json')}")
            with open(os.path.join(scaler_save_dir, 'ciciot_label_mapping.json'), 'w') as f:
                json.dump(new_label_to_int, f, indent=2)
            print(f"DEBUG: Label mapping saved")
    else:
        # Original behavior: use existing scaler and optionally development subset
        # Convert artifact_dir to absolute path relative to project root (already done above)
        # Also convert data_dir and development_subset_path if they're relative (already done above)

        # Create datasets (withhold only from train for true open-set)
        train_dataset = CICIoT2023ProtoIDSDataset(
            data_dir=data_dir,
            split='train',
            artifact_dir=artifact_dir,
            development_subset_path=development_subset_path,
            withheld_classes=withheld_classes
        )

        val_dataset = CICIoT2023ProtoIDSDataset(
            data_dir=data_dir,
            split='validation',
            artifact_dir=artifact_dir
        )

        test_dataset = CICIoT2023ProtoIDSDataset(
            data_dir=data_dir,
            split='test',
            artifact_dir=artifact_dir
        )

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
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Get metadata
    if open_set:
        # For open_set=True, we have computed new_label_to_int
        # Number of classes = number of known classes + 1 (for unknown class)
        num_classes = len(new_label_to_int) + 1
        input_dim = len(feature_cols)
    else:
        # For open_set=False, use the dataset methods
        num_classes = train_dataset.get_num_classes()
        input_dim = train_dataset.get_input_dim()

    return train_loader, val_loader, test_loader, num_classes, input_dim