"""
Preprocessing pipeline for Edge-IIoTset dataset.
Only builds and tests the preprocessing pipeline, no model training.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import joblib

def load_edgeiiot_data(filepath):
    """
    Load the Edge-IIoT dataset and convert all columns to numeric, coercing errors.
    This handles cases where numeric columns contain strings (e.g., error messages).
    """
    df = pd.read_csv(filepath, low_memory=False)  # Avoid mixed type warning
    # Convert all columns to numeric, errors='coerce' will turn non-numeric to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_columns_to_remove_edgeiiot():
    """
    Return columns to remove for Edge-IIoT based on forensic analysis.
    High-risk leakage columns and constant columns.
    """
    # Label columns (to be separated)
    label_cols = ['Attack_label', 'Attack_type']

    # High-risk identifier columns (leakage)
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

    # Additional columns recommended to remove in the report:
    additional_cols = [
        'mqtt.conack.flags',
        'mqtt.protoname',
        'mqtt.topic',
        'http.referer'
    ]

    cols_to_remove = label_cols + high_risk_cols + additional_cols

    reasons = {}
    for col in label_cols:
        reasons[col] = 'Label column'
    for col in high_risk_cols:
        reasons[col] = 'High leakage risk (IP, timestamp, port, payload)'
    for col in additional_cols:
        reasons[col] = 'Recommended for removal in forensic report (environment-specific or leakage)'

    return cols_to_remove, reasons

def get_feature_columns(df, cols_to_remove):
    """
    Return list of feature column names after removing specified columns.
    """
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in cols_to_remove]
    return feature_cols

def remove_constant_columns(df, feature_cols, verbose=False):
    """
    Remove constant columns (only one unique value) from the feature set.
    Returns the list of non-constant feature columns.
    """
    non_constant_cols = []
    constant_cols = []
    for col in feature_cols:
        # Check if the column is constant (after ignoring NaN?)
        # We'll consider a column constant if it has only one unique value (excluding NaN)
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 1:
            constant_cols.append(col)
        else:
            non_constant_cols.append(col)
    if verbose:
        print(f"Constant columns found: {constant_cols}")
    return non_constant_cols, constant_cols

def preprocess_edgeiiot(df, feature_cols):
    """
    Preprocess the dataframe:
    - Separate features and labels
    - Fill NaN with 0 (after conversion, NaN may appear from non-numeric strings)
    - Fit StandardScaler on the data (we'll assume the whole dataset is for preprocessing only)
    """
    # Select features
    X = df[feature_cols].copy()

    # Fill NaN with 0 (since we coerced errors to NaN)
    X = X.fillna(0)

    # Initialize scaler
    scaler = StandardScaler()

    # Fit on the data (in a real scenario, this would be training data only)
    X_scaled = scaler.fit_transform(X)

    # Convert to float32
    X_scaled = X_scaled.astype(np.float32)

    return X_scaled, scaler

def save_preprocessing_artifacts(scaler, feature_cols, original_cols, removed_cols, reasons, output_dir):
    """
    Save scaler and feature manifest.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save scaler
    scaler_path = os.path.join(output_dir, 'edgeiiot_preprocessor.joblib')
    joblib.dump(scaler, scaler_path)

    # Create feature manifest
    manifest = {
        'original_columns': original_cols,
        'removed_columns': removed_cols,
        'retained_columns': feature_cols,
        'removal_reasons': reasons
    }
    manifest_path = os.path.join(output_dir, 'edgeiiot_feature_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return scaler_path, manifest_path

def main():
    """
    Main preprocessing pipeline for Edge-IIoTset.
    """
    print("Starting Edge-IIoTset preprocessing...")

    # Paths
    data_path = 'DNN-EdgeIIoT-dataset.csv/DNN-EdgeIIoT-dataset.csv'
    output_dir = 'results/baselines'

    # Load data
    print("Loading Edge-IIoT dataset...")
    df = load_edgeiiot_data(data_path)
    print(f"Dataset shape: {df.shape}")

    # Get columns to remove (based on forensic analysis)
    cols_to_remove, reasons = get_columns_to_remove_edgeiiot()
    print(f"Number of columns to remove initially: {len(cols_to_remove)}")

    # Get feature columns after removing the specified ones
    feature_cols = get_feature_columns(df, cols_to_remove)
    print(f"Number of features after initial removal: {len(feature_cols)}")

    # Remove constant columns from the feature set
    print("Checking for constant columns...")
    non_constant_cols, constant_cols = remove_constant_columns(df, feature_cols, verbose=True)
    # Update the feature columns and the removed columns list
    feature_cols = non_constant_cols
    # Add the constant columns to the removed columns list
    cols_to_remove_extended = cols_to_remove + constant_cols
    # Update reasons for constant columns
    for col in constant_cols:
        reasons[col] = 'Constant column (only one unique value)'

    print(f"Number of features after removing constant columns: {len(feature_cols)}")

    # Preprocess (scale) the features
    print("Scaling features...")
    X_scaled, scaler = preprocess_edgeiiot(df, feature_cols)
    print(f"Shape of scaled features: {X_scaled.shape}")

    # Save preprocessing artifacts
    print("Saving preprocessing artifacts...")
    original_columns = df.columns.tolist()
    scaler_path, manifest_path = save_preprocessing_artifacts(
        scaler, feature_cols, original_columns, cols_to_remove_extended, reasons, output_dir
    )

    # Optionally, save a small sample of the preprocessed data for inspection
    sample_dir = 'experiments/data'
    os.makedirs(sample_dir, exist_ok=True)
    # We'll save the first 1000 rows as a CSV for quick inspection
    sample_scaled_path = os.path.join(sample_dir, 'edgeiiot_sample_scaled.npy')
    np.save(sample_scaled_path, X_scaled[:1000])
    print(f"Saved sample of scaled features (first 1000 rows) to {sample_scaled_path}")

    print("Preprocessing complete!")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved feature manifest to: {manifest_path}")

if __name__ == '__main__':
    main()