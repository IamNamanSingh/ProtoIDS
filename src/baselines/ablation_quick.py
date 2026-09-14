"""
Quick ablation study on feature selection for CICIoT2023 dataset.
Compares Experiment A (all retained features) vs Experiment B (without near-constant features).
Uses Random Forest with 10 trees for speed.
Fixed preprocessing to ensure consistency between experiments.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_preprocessing_artifacts(artifact_dir):
    """
    Load scaler, feature manifest, and label mapping.
    """
    scaler_path = os.path.join(artifact_dir, 'ciciot_preprocessor.joblib')
    manifest_path = os.path.join(artifact_dir, 'ciciot_feature_manifest.json')
    label_mapping_path = os.path.join(artifact_dir, 'ciciot_label_mapping.json')

    scaler = joblib.load(scaler_path)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    return scaler, manifest, label_mapping


def load_development_subset(dev_parquet_path, feature_cols):
    """
    Load the development subset Parquet file and return features and labels.
    """
    dev_df = pd.read_parquet(dev_parquet_path)
    # Select only the desired feature columns
    X_dev = dev_df[feature_cols].copy().values.astype(np.float32)
    y_multi_dev = dev_df['label_multi'].copy().values
    y_bin_dev = dev_df['label_binary'].copy().values
    return X_dev, y_multi_dev, y_bin_dev


def load_and_preprocess_split(split_path, feature_cols, label_col='label'):
    """
    Load a CSV split and return features and label.
    """
    df = pd.read_csv(split_path)
    # Ensure we only keep the feature columns (in case the order is different)
    X = df[feature_cols].copy()
    y_multi = df[label_col].copy()
    # Create binary label
    y_bin = y_multi.apply(lambda x: 0 if x == 'BenignTraffic' else 1)
    return X, y_multi, y_bin


def fit_scaler_and_transform(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data and transform all splits.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to float32
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_val_scaled = X_val_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def compute_metrics(y_true, y_pred, prefix=''):
    """
    Compute classification metrics.
    """
    metrics = {}

    # Basic metrics
    metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{prefix}precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics[f'{prefix}mcc'] = matthews_corrcoef(y_true, y_pred)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Store per-class metrics as lists
    metrics[f'{prefix}precision_per_class'] = precision_per_class.tolist()
    metrics[f'{prefix}recall_per_class'] = recall_per_class.tolist()
    metrics[f'{prefix}f1_per_class'] = f1_per_class.tolist()

    return metrics


def main():
    print("Starting corrected quick ablation study...")

    # Paths
    artifact_dir = 'results/baselines'
    data_dir = 'CICIOT23'
    dev_parquet_path = 'experiments/data/ciciot_dev.parquet'
    results_dir = 'results/baselines'
    os.makedirs(results_dir, exist_ok=True)

    # Load preprocessing artifacts
    print("Loading preprocessing artifacts...")
    _, manifest, label_mapping = load_preprocessing_artifacts(artifact_dir)
    feature_cols_all = manifest['retained_columns']  # 41 features
    print(f"Number of features in full set: {len(feature_cols_all)}")

    # Define the near-constant features to remove for Experiment B
    near_constant_to_remove = ['ece_flag_number', 'cwr_flag_number', 'DNS', 'SSH', 'DHCP', 'ARP', 'IPv']
    feature_cols_exp_b = [f for f in feature_cols_all if f not in near_constant_to_remove]
    print(f"Number of features in Experiment B: {len(feature_cols_exp_b)}")
    print(f"Removed features: {near_constant_to_remove}")

    # Load development subset for both experiments
    print("Loading development subset...")
    X_dev_all, y_multi_dev_all, _ = load_development_subset(dev_parquet_path, feature_cols_all)
    X_dev_b, y_multi_dev_b, _ = load_development_subset(dev_parquet_path, feature_cols_exp_b)
    print(f"Development subset shape (all features): {X_dev_all.shape}")
    print(f"Development subset shape (exp B): {X_dev_b.shape}")

    # Verify that labels are the same for both feature sets (should be)
    if not np.array_equal(y_multi_dev_all, y_multi_dev_b):
        print("WARNING: Labels differ between feature sets!")
    else:
        print("Labels are consistent between feature sets.")

    # Load and preprocess validation and test sets
    print("Loading and preprocessing validation and test sets...")
    X_val_all_raw, y_val_multi_all, _ = load_and_preprocess_split(
        os.path.join(data_dir, 'validation', 'validation.csv'),
        feature_cols_all
    )
    X_test_all_raw, y_test_multi_all, _ = load_and_preprocess_split(
        os.path.join(data_dir, 'test', 'test.csv'),
        feature_cols_all
    )
    X_val_b_raw, y_val_multi_b, _ = load_and_preprocess_split(
        os.path.join(data_dir, 'validation', 'validation.csv'),
        feature_cols_exp_b
    )
    X_test_b_raw, y_test_multi_b, _ = load_and_preprocess_split(
        os.path.join(data_dir, 'test', 'test.csv'),
        feature_cols_exp_b
    )

    # Convert labels to integer for multiclass
    label_to_int = label_mapping['multiclass']['label_to_int']
    y_multi_dev_int_all = np.array([label_to_int[label] for label in y_multi_dev_all])
    y_multi_dev_int_b = np.array([label_to_int[label] for label in y_multi_dev_b])
    y_val_multi_int_all = np.array([label_to_int[label] for label in y_val_multi_all])
    y_val_multi_int_b = np.array([label_to_int[label] for label in y_val_multi_b])
    y_test_multi_int_all = np.array([label_to_int[label] for label in y_test_multi_all])
    y_test_multi_int_b = np.array([label_to_int[label] for label in y_test_multi_b])

    # Experiment A: all features
    print("\nPreprocessing validation and test sets for Experiment A...")
    scaler_a = StandardScaler()
    X_train_a = scaler_a.fit_transform(X_dev_all).astype(np.float32)
    X_val_all = scaler_a.transform(X_val_all_raw).astype(np.float32)
    X_test_all = scaler_a.transform(X_test_all_raw).astype(np.float32)

    # Train and evaluate Random Forest for Experiment A (all features)
    print("\nTraining Random Forest on Experiment A (all features) with 10 trees...")
    start_time = time.time()
    rf_a = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf_a.fit(X_train_a, y_multi_dev_int_all)
    train_time_a = time.time() - start_time

    # Predict on validation
    start_time = time.time()
    y_val_pred_a = rf_a.predict(X_val_all)
    pred_time_a = time.time() - start_time

    # Compute metrics
    metrics_a = compute_metrics(y_val_multi_int_all, y_val_pred_a, prefix='val_')
    metrics_a.update({
        'train_time': train_time_a,
        'pred_time': pred_time_a,
        'experiment': 'A',
        'feature_set': 'all_retained',
        'n_features': len(feature_cols_all)
    })
    print(f"Experiment A - Validation Accuracy: {metrics_a['val_accuracy']:.4f}")
    print(f"Experiment A - Validation Macro F1: {metrics_a['val_f1_macro']:.4f}")

    # Experiment B: without near-constant features
    print("\nPreprocessing validation and test sets for Experiment B...")
    scaler_b = StandardScaler()
    X_train_b = scaler_b.fit_transform(X_dev_b).astype(np.float32)
    X_val_b = scaler_b.transform(X_val_b_raw).astype(np.float32)
    X_test_b = scaler_b.transform(X_test_b_raw).astype(np.float32)

    # Train and evaluate Random Forest for Experiment B (without near-constant features)
    print("\nTraining Random Forest on Experiment B (without near-constant features) with 10 trees...")
    start_time = time.time()
    rf_b = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf_b.fit(X_train_b, y_multi_dev_int_b)
    train_time_b = time.time() - start_time

    # Predict on validation
    start_time = time.time()
    y_val_pred_b = rf_b.predict(X_val_b)
    pred_time_b = time.time() - start_time

    # Compute metrics
    metrics_b = compute_metrics(y_val_multi_int_b, y_val_pred_b, prefix='val_')
    metrics_b.update({
        'train_time': train_time_b,
        'pred_time': pred_time_b,
        'experiment': 'B',
        'feature_set': 'without_near_constant',
        'n_features': len(feature_cols_exp_b)
    })
    print(f"Experiment B - Validation Accuracy: {metrics_b['val_accuracy']:.4f}")
    print(f"Experiment B - Validation Macro F1: {metrics_b['val_f1_macro']:.4f}")

    # Compare the results
    print("\n=== Ablation Comparison ===")
    print(f"{'Metric':<25} {'Experiment A':<15} {'Experiment B':<15} {'Difference (B-A)':<15}")
    print("-" * 70)
    for metric in ['val_accuracy', 'val_precision_macro', 'val_recall_macro', 'val_f1_macro', 'val_mcc']:
        val_a = metrics_a.get(metric, 0)
        val_b = metrics_b.get(metric, 0)
        diff = val_b - val_a
        print(f"{metric:<25} {val_a:<15.4f} {val_b:<15.4f} {diff:<15.4f}")

    # Also look at minority-class recall (recall for rare classes)
    # We'll compute the average recall for the top 5 rare classes (by frequency in development set)
    class_counts = pd.Series(y_multi_dev_all).value_counts()
    # Get the 5 rarest classes (smallest counts)
    rare_classes = class_counts.tail(5).index.tolist()
    print(f"\nRare classes in development set: {rare_classes}")
    # Get the indices of these classes in the label mapping
    label_to_int = label_mapping['multiclass']['label_to_int']
    rare_class_indices = [label_to_int[cls] for cls in rare_classes if cls in label_to_int]
    # Compute recall for these classes from the per-class recall arrays
    recall_per_class_a = metrics_a['val_recall_per_class']
    recall_per_class_b = metrics_b['val_recall_per_class']
    # Average recall for rare classes
    rare_recall_a = np.mean([recall_per_class_a[i] for i in rare_class_indices if i < len(recall_per_class_a)])
    rare_recall_b = np.mean([recall_per_class_b[i] for i in rare_class_indices if i < len(recall_per_class_b)])
    print(f"Average recall for rare classes - Experiment A: {rare_recall_a:.4f}")
    print(f"Average recall for rare classes - Experiment B: {rare_recall_b:.4f}")
    print(f"Difference (B-A): {rare_recall_b - rare_recall_a:.4f}")

    # Save ablation results
    ablation_results = {
        'experiment_A': metrics_a,
        'experiment_B': metrics_b,
        'rare_classes': rare_classes,
        'rare_recall_A': float(rare_recall_a),
        'rare_recall_B': float(rare_recall_b)
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        else:
            return obj

    ablation_results_json = convert_for_json(ablation_results)

    results_path = os.path.join(results_dir, 'ablation_results_corrected.json')
    with open(results_path, 'w') as f:
        json.dump(ablation_results_json, f, indent=2)

    print(f"\nCorrected ablation results saved to {results_path}")

if __name__ == '__main__':
    main()