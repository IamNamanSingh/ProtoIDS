"""
Run baseline experiments on CICIoT2023 dataset.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set joblib backend to avoid loky issues on Windows
os.environ['JOBLIB_START_METHOD'] = 'spawn'
# Also try to set n_jobs=1 by default in sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, average_precision_score
)
import joblib

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will use HistGradientBoostingClassifier for gradient boosting baseline.")

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

def load_and_preprocess_split(split_path, scaler, feature_cols, label_col='label'):
    """
    Load a CSV split and preprocess it:
    - Remove columns not in feature_cols
    - Separate features and label
    - Apply scaling
    """
    df = pd.read_csv(split_path)

    # Ensure we only keep the feature columns (in case the order is different)
    X = df[feature_cols].copy()
    y_multi = df[label_col].copy()

    # Create binary label
    y_bin = y_multi.apply(lambda x: 0 if x == 'BenignTraffic' else 1)

    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = X_scaled.astype(np.float32)

    return X_scaled, y_multi, y_bin

def get_development_subset(dev_parquet_path):
    """
    Load the development subset Parquet file.
    """
    dev_df = pd.read_parquet(dev_parquet_path)
    feature_cols = [c for c in dev_df.columns if c not in ['label_multi', 'label_binary']]
    X_dev = dev_df[feature_cols].values.astype(np.float32)
    y_multi_dev = dev_df['label_multi'].values
    y_bin_dev = dev_df['label_binary'].values
    return X_dev, y_multi_dev, y_bin_dev, feature_cols

def compute_metrics(y_true, y_pred, y_pred_proba=None, prefix=''):
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

    # Store per-class metrics as lists (or we can store as dict with class names)
    # We'll store as lists and later map to class names if needed.
    metrics[f'{prefix}precision_per_class'] = precision_per_class.tolist()
    metrics[f'{prefix}recall_per_class'] = recall_per_class.tolist()
    metrics[f'{prefix}f1_per_class'] = f1_per_class.tolist()

    # If probabilities are provided, compute ROC-AUC and PR-AUC for binary case
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        # For binary, we need the probability of the positive class
        if y_pred_proba.shape[1] == 2:
            y_pred_proba_pos = y_pred_proba[:, 1]
        else:
            # If only one class is present in predictions, we cannot compute
            y_pred_proba_pos = None

        if y_pred_proba_pos is not None:
            try:
                metrics[f'{prefix}roc_auc'] = roc_auc_score(y_true, y_pred_proba_pos)
                metrics[f'{prefix}pr_auc'] = average_precision_score(y_true, y_pred_proba_pos)
            except ValueError:
                # In case of only one class in y_true
                metrics[f'{prefix}roc_auc'] = 0.0
                metrics[f'{prefix}pr_auc'] = 0.0

    return metrics

def main():
    print("Starting baseline experiments...")

    # Paths
    artifact_dir = 'results/baselines'
    data_dir = 'CICIOT23'
    dev_parquet_path = 'experiments/data/ciciot_dev.parquet'
    results_dir = 'results/baselines'
    os.makedirs(results_dir, exist_ok=True)

    # Load preprocessing artifacts
    print("Loading preprocessing artifacts...")
    scaler, manifest, label_mapping = load_preprocessing_artifacts(artifact_dir)
    feature_cols = manifest['retained_columns']
    print(f"Number of features: {len(feature_cols)}")

    # Load development subset for training
    print("Loading development subset...")
    X_train_dev, y_train_multi_dev, y_train_bin_dev, _ = get_development_subset(dev_parquet_path)
    print(f"Development subset shape: {X_train_dev.shape}")

    # Load and preprocess validation and test sets
    print("Loading and preprocessing validation set...")
    X_val, y_val_multi, y_val_bin = load_and_preprocess_split(
        os.path.join(data_dir, 'validation', 'validation.csv'),
        scaler, feature_cols
    )
    print(f"Validation shape: {X_val.shape}")

    print("Loading and preprocessing test set...")
    X_test, y_test_multi, y_test_bin = load_and_preprocess_split(
        os.path.join(data_dir, 'test', 'test.csv'),
        scaler, feature_cols
    )
    print(f"Test shape: {X_test.shape}")

    # Convert labels to integer for multiclass (using the label mapping)
    label_to_int = label_mapping['multiclass']['label_to_int']
    y_train_multi_dev_int = np.array([label_to_int[label] for label in y_train_multi_dev])
    y_val_multi_int = np.array([label_to_int[label] for label in y_val_multi])
    y_test_multi_int = np.array([label_to_int[label] for label in y_test_multi])

    # Initialize results dictionary
    all_results = {}

    # ====================================================
    # Baseline A: Random Forest
    # ====================================================
    print("\nTraining Random Forest...")
    start_time = time.time()
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1  # Set to 1 to avoid joblib issues
    )
    try:
        rf_classifier.fit(X_train_dev, y_train_multi_dev_int)
        rf_train_time = time.time() - start_time
        print("Random Forest fit successful")
    except Exception as e:
        print(f"Error fitting Random Forest: {e}")
        rf_train_time = time.time() - start_time
        # Create dummy results to avoid breaking the script
        rf_classifier = None

    if rf_classifier is not None:
        # Predict on validation
        start_time = time.time()
        y_val_pred_rf = rf_classifier.predict(X_val)
        y_val_pred_proba_rf = rf_classifier.predict_proba(X_val)
        rf_pred_time = time.time() - start_time

        # Compute metrics
        rf_results = compute_metrics(y_val_multi_int, y_val_pred_rf, y_val_pred_proba_rf, prefix='val_')
        rf_results.update({
            'train_time': rf_train_time,
            'pred_time': rf_pred_time,
            'model_type': 'RandomForest'
        })
        all_results['random_forest'] = rf_results

        # Also compute on development set for reference
        y_train_pred_rf = rf_classifier.predict(X_train_dev)
        y_train_pred_proba_rf = rf_classifier.predict_proba(X_train_dev)
        train_metrics_rf = compute_metrics(y_train_multi_dev_int, y_train_pred_rf, y_train_pred_proba_rf, prefix='train_')
        rf_results.update(train_metrics_rf)

        print(f"Random Forest validation accuracy: {rf_results['val_accuracy']:.4f}")
        print(f"Random Forest validation macro F1: {rf_results['val_f1_macro']:.4f}")
    else:
        print("Skipping Random Forest due to fit error")

    # ====================================================
    # Baseline B: Gradient Boosting
    # ====================================================
    print("\nTraining Gradient Boosting...")
    start_time = time.time()
    if XGBOOST_AVAILABLE:
        gb_classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=1
        )
    else:
        gb_classifier = HistGradientBoostingClassifier(
            max_iter=100,
            random_state=42
        )
    try:
        gb_classifier.fit(X_train_dev, y_train_multi_dev_int)
        gb_train_time = time.time() - start_time
        print("Gradient Boosting fit successful")
    except Exception as e:
        print(f"Error fitting Gradient Boosting: {e}")
        gb_train_time = time.time() - start_time
        gb_classifier = None

    if gb_classifier is not None:
        # Predict on validation
        start_time = time.time()
        y_val_pred_gb = gb_classifier.predict(X_val)
        if hasattr(gb_classifier, 'predict_proba'):
            y_val_pred_proba_gb = gb_classifier.predict_proba(X_val)
        else:
            y_val_pred_proba_gb = gb_classifier.predict_proba(X_val)
        gb_pred_time = time.time() - start_time

        # Compute metrics
        gb_results = compute_metrics(y_val_multi_int, y_val_pred_gb, y_val_pred_proba_gb, prefix='val_')
        gb_results.update({
            'train_time': gb_train_time,
            'pred_time': gb_pred_time,
            'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'HistGradientBoosting'
        })
        all_results['gradient_boosting'] = gb_results

        # Also compute on development set
        y_train_pred_gb = gb_classifier.predict(X_train_dev)
        y_train_pred_proba_gb = gb_classifier.predict_proba(X_train_dev)
        train_metrics_gb = compute_metrics(y_train_multi_dev_int, y_train_pred_gb, y_train_pred_proba_gb, prefix='train_')
        gb_results.update(train_metrics_gb)

        print(f"Gradient Boosting validation accuracy: {gb_results['val_accuracy']:.4f}")
        print(f"Gradient Boosting validation macro F1: {gb_results['val_f1_macro']:.4f}")
    else:
        print("Skipping Gradient Boosting due to fit error")

    # ====================================================
    # Baseline C: Simple MLP
    # ====================================================
    print("\nTraining Simple MLP...")
    # We'll use a simple MLP with two hidden layers, ReLU, dropout.
    # Since we don't want to add heavy dependencies, we'll use sklearn's MLPClassifier.
    # Note: sklearn's MLPClassifier uses LBFGS or Adam, and we can tune hidden layers.
    from sklearn.neural_network import MLPClassifier

    start_time = time.time()
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate='adaptive',
        max_iter=20,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    try:
        mlp_classifier.fit(X_train_dev, y_train_multi_dev_int)
        mlp_train_time = time.time() - start_time
        print("MLP fit successful")
    except Exception as e:
        print(f"Error fitting MLP: {e}")
        mlp_train_time = time.time() - start_time
        mlp_classifier = None

    if mlp_classifier is not None:
        # Predict on validation
        start_time = time.time()
        y_val_pred_mlp = mlp_classifier.predict(X_val)
        y_val_pred_proba_mlp = mlp_classifier.predict_proba(X_val)
        mlp_pred_time = time.time() - start_time

        # Compute metrics
        mlp_results = compute_metrics(y_val_multi_int, y_val_pred_mlp, y_val_pred_proba_mlp, prefix='val_')
        mlp_results.update({
            'train_time': mlp_train_time,
            'pred_time': mlp_pred_time,
            'model_type': 'MLP'
        })
        all_results['mlp'] = mlp_results

        # Also compute on development set
        y_train_pred_mlp = mlp_classifier.predict(X_train_dev)
        y_train_pred_proba_mlp = mlp_classifier.predict_proba(X_train_dev)
        train_metrics_mlp = compute_metrics(y_train_multi_dev_int, y_train_pred_mlp, y_train_pred_proba_mlp, prefix='train_')
        mlp_results.update(train_metrics_mlp)

        print(f"MLP validation accuracy: {mlp_results['val_accuracy']:.4f}")
        print(f"MLP validation macro F1: {mlp_results['val_f1_macro']:.4f}")
    else:
        print("Skipping MLP due to fit error")

    # ====================================================
    # Save results
    # ====================================================
    results_path = os.path.join(results_dir, 'baseline_experiment_results.json')
    # Convert numpy arrays to lists for JSON serialization
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

    all_results_json = convert_for_json(all_results)

    with open(results_path, 'w') as f:
        json.dump(all_results_json, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for model_name, results in all_results.items():
        print(f"{model_name}:")
        print(f"  Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"  Validation Macro F1: {results['val_f1_macro']:.4f}")
        print(f"  Validation MCC: {results['val_mcc']:.4f}")
        print()

if __name__ == '__main__':
    main()