"""
ProtoIDS Dataset Module
Handles loading and preprocessing of CICIoT2023 data for ProtoIDS.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class CICIoT2023ProtoIDSDataset(Dataset):
    """
    Dataset class for CICIoT2023 data with preprocessing for ProtoIDS.
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 artifact_dir: str = 'results/baselines',
                 development_subset_path: Optional[str] = None):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing the CSV splits (train/validation/test)
            split: Which split to load ('train', 'validation', or 'test')
            artifact_dir: Directory containing preprocessing artifacts
            development_subset_path: Path to development subset Parquet (for train split only)
        """
        self.data_dir = data_dir
        self.split = split
        self.artifact_dir = artifact_dir

        # Load preprocessing artifacts
        self.scaler, self.manifest, self.label_mapping = self._load_artifacts()

        # Get feature columns
        self.feature_cols = self.manifest['retained_columns']
        self.label_to_int = self.label_mapping['multiclass']['label_to_int']
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

        # Load data
        self.X, self.y_multi, self.y_bin = self._load_data(
            development_subset_path if split == 'train' else None
        )

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
                       val_batch_size: Optional[int] = None):
    """
    Create data loaders for training, validation, and test sets.

    Args:
        data_dir: Directory containing the CSV splits
        batch_size: Batch size for training
        artifact_dir: Directory containing preprocessing artifacts
        development_subset_path: Path to development subset Parquet
        val_batch_size: Batch size for validation/test (defaults to batch_size)

    Returns:
        tuple: (train_loader, val_loader, test_loader, num_classes, input_dim)
    """
    if val_batch_size is None:
        val_batch_size = batch_size

    # Convert artifact_dir to absolute path relative to project root
    if not os.path.isabs(artifact_dir):
        # Get project root (assuming this file is in src/protoids/dataset.py)
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))  # src/protoids/dataset.py -> src/protoids -> src
        project_root = os.path.dirname(src_dir)  # src -> project root
        artifact_dir = os.path.join(project_root, artifact_dir)

        # Also convert data_dir and development_subset_path if they're relative
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir)
        if not os.path.isabs(development_subset_path):
            development_subset_path = os.path.join(project_root, development_subset_path)

    # Create datasets
    train_dataset = CICIoT2023ProtoIDSDataset(
        data_dir=data_dir,
        split='train',
        artifact_dir=artifact_dir,
        development_subset_path=development_subset_path
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
    num_classes = train_dataset.get_num_classes()
    input_dim = train_dataset.get_input_dim()

    return train_loader, val_loader, test_loader, num_classes, input_dim