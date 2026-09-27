"""
Ablation study on feature selection for CICIoT2023 dataset.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_split_raw(split_path, feature_cols, label_col='label'):
    """
    Load a CSV split and return raw features and label (not scaled).
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

def get_development_subset_raw(dev_parquet_path, feature_cols):
    """
    Load the development subset Parquet file and return raw features and labels.
    """
    dev_df = pd.read_parquet(dev_parquet_path)
    # Select only the desired feature columns
    X_dev = dev_df[feature_cols].copy()
    y_multi_dev = dev_df['label_multi'].copy()
    y_bin_dev = dev_df['label_binary'].copy()
    return X_dev, y_multi_dev, y_bin_dev

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
    print("Starting ablation study...")

    # Paths
    data_dir = 'CICIOT23'
    dev_parquet_path = 'experiments/data/ciciot_dev.parquet'
    results_dir = 'results/baselines'
    os.makedirs(results_dir, exist_ok=True)

    # Load the manifest to get the original feature columns
    manifest_path = 'results/baselines/ciciot_feature_manifest.json'
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    feature_cols_all = manifest['retained_columns']  # 41 features
    print(f"Number of features in full set: {len(feature_cols_all)}")

    # Define the near-constant features to remove for Experiment B
    near_constant_to_remove = ['ece_flag_number', 'cwr_flag_number', 'DNS', 'SSH', 'DHCP', 'ARP', 'IPv']
    feature_cols_exp_b = [f for f in feature_cols_all if f not in near_constant_to_remove]
    print(f"Number of features in Experiment B: {len(feature_cols_exp_b)}")
    print(f"Removed features: {near_constant_to_remove}")

    # Load development subset (already scaled via saved scaler)
    print("Loading development subset (already scaled)...")
    scaler = joblib.load('results/baselines/ciciot_preprocessor.joblib')
    X_dev_all_df, y_multi_dev, y_bin_dev = get_development_subset_raw(dev_parquet_path, feature_cols_all)
    X_dev_b_df, _, _ = get_development_subset_raw(dev_parquet_path, feature_cols_exp_b)
    # Convert to float32 - already scaled
    X_train_all_scaled = X_dev_all_df.values.astype(np.float32) if hasattr(X_dev_all_df, 'values') else np.array(X_dev_all_df).astype(np.float32)
    X_train_b_scaled = X_dev_b_df.values.astype(np.float32) if hasattr(X_dev_b_df, 'values') else np.array(X_dev_b_df).astype(np.float32)
    print(f"Development subset shape (all features): {X_train_all_scaled.shape}")
    print(f"Development subset shape (exp B): {X_train_b_scaled.shape}")
    print("  Using pre-fitted scaler; dev set not refit (fixes double-scaling bug)")

    # Load and preprocess validation and test sets for Experiment A
    print("Loading and preprocessing validation set for Experiment A...")
    X_val_all_raw, y_val_multi, y_val_bin = load_and_preprocess_split_raw(
        os.path.join(data_dir, 'validation', 'validation.csv'),
        feature_cols_all
    )
    X_test_all_raw, y_test_multi, y_test_bin = load_and_preprocess_split_raw(
        os.path.join(data_dir, 'test', 'test.csv'),
        feature_cols_all
    )
    X_val_all_scaled = scaler.transform(X_val_all_raw).astype(np.float32)
    X_test_all_scaled = scaler.transform(X_test_all_raw).astype(np.float32)

    # Load validation/test for Experiment B - derive via subsetting the scaled full validation
    print("Loading validation set for Experiment B (via subsetting full scaled matrix)...")
    col_to_idx = {col: i for i, col in enumerate(feature_cols_all)}
    keep_idx = [col_to_idx[c] for c in feature_cols_exp_b]
    X_val_b_scaled = X_val_all_scaled[:, keep_idx]
    X_test_b_scaled = X_test_all_scaled[:, keep_idx]

    # Convert labels to integer for multiclass
    label_mapping_path = 'results/baselines/ciciot_label_mapping.json'
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    label_to_int = label_mapping['multiclass']['label_to_int']
    y_multi_dev_int = np.array([label_to_int[label] for label in y_multi_dev])
    y_val_multi_int = np.array([label_to_int[label] for label in y_val_multi])
    y_test_multi_int = np.array([label_to_int[label] for label in y_test_multi])

    # Train and evaluate Random Forest for Experiment A (all features)
    print("\nTraining Random Forest on Experiment A (all features)...")
    start_time = time.time()
    rf_a = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_a.fit(X_train_all_scaled, y_multi_dev_int)
    train_time_a = time.time() - start_time

    # Predict on validation
    start_time = time.time()
    y_val_pred_a = rf_a.predict(X_val_all_scaled)
    pred_time_a = time.time() - start_time

    # Compute metrics
    metrics_a = compute_metrics(y_val_multi_int, y_val_pred_a, prefix='val_')
    metrics_a.update({
        'train_time': train_time_a,
        'pred_time': pred_time_a,
        'experiment': 'A',
        'feature_set': 'all_retained',
        'n_features': len(feature_cols_all)
    })
    print(f"Experiment A - Validation Accuracy: {metrics_a['val_accuracy']:.4f}")
    print(f"Experiment A - Validation Macro F1: {metrics_a['val_f1_macro']:.4f}")

    # Train and evaluate Random Forest for Experiment B (without near-constant features)
    print("\nTraining Random Forest on Experiment B (without near-constant features)...")
    start_time = time.time()
    rf_b = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    rf_b.fit(X_train_b_scaled, y_multi_dev_int)
    train_time_b = time.time() - start_time

    # Predict on validation
    start_time = time.time()
    y_val_pred_b = rf_b.predict(X_val_b_scaled)
    pred_time_b = time.time() - start_time

    # Compute metrics
    metrics_b = compute_metrics(y_val_multi_int, y_val_pred_b, prefix='val_')
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
    class_counts = pd.Series(y_multi_dev).value_counts()
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

    results_path = os.path.join(results_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(ablation_results_json, f, indent=2)

    print(f"\nAblation results saved to {results_path}")

if __name__ == '__main__':
    main()