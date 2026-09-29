import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Set environment variable for data directory
# Expected: directory containing the CSV file (parent of DNN-EdgeIIoT-dataset.csv)
data_dir = os.environ.get('EDGEIIOT_DATA_DIR')
if not data_dir:
    # Fallback to typical location
    data_dir = r'C:\Users\LENOVO\Desktop\workspace\Capstone Project\DNN-EdgeIIoT-dataset.csv'
    os.environ['EDGEIIOT_DATA_DIR'] = data_dir
    print(f"Set EDGEIIOT_DATA_DIR to: {data_dir}")

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
print(f"File size: {os.path.getsize(csv_path) / (1024*1024):.2f} MB")

# Load CSV, coerce all columns except label columns to numeric, fill NaN with 0
df_raw = pd.read_csv(csv_path, low_memory=False)
print(f"Raw shape: {df_raw.shape}")
original_dtypes = df_raw.dtypes.to_dict()

# We will coerce only feature columns (all except Attack_type and Attack_label)
label_cols_for_coercion = ['Attack_type', 'Attack_label']
df = df_raw.copy()
# Identify columns to coerce: all columns except label columns
cols_to_coerce = [col for col in df.columns if col not in label_cols_for_coercion]
coerced_cols = []
for col in cols_to_coerce:
    original_col = df[col].copy()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Identify which values became NaN due to coercion
    if df[col].isna().any():
        coerced_cols.append(col)
    # Fill NaN with 0 after coercion
    df[col] = df[col].fillna(0)
if coerced_cols:
    warnings.warn(f"Columns required coercion from non-numeric to numeric (NaN replaced with 0): {coerced_cols}")
    print(f"Number of coerced columns: {len(coerced_cols)}")
    # Show first few
    for col in coerced_cols[:5]:
        print(f"  - {col}")

# Note: label columns (Attack_type, Attack_label) remain as original strings

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
print(f"\nNumber of features after removing leakage and label columns: {len(feature_cols)}")

# Prepare label mapping for Attack_type
# We'll get unique Attack_type values from the original raw data (as strings)
attack_type_series = df_raw['Attack_type']
unique_attack_types = sorted(attack_type_series.unique())
print(f"Unique Attack_type values ({len(unique_attack_types)}): {unique_attack_types}")

# Determine which classes are withheld (for true open-set experiment)
withheld_classes = ['MITM', 'Password', 'Vulnerability_scanner']
print(f"\nWithheld classes (to be mapped to UNKNOWN): {withheld_classes}")

# Validate that all withheld classes exist in the dataset
withheld_set = set(withheld_classes)
unknown_in_data = withheld_set - set(unique_attack_types)
if unknown_in_data:
    raise ValueError(f"The following withheld classes are not present in the dataset: {unknown_in_data}")
known_set = set(unique_attack_types) - withheld_set
print(f"Known classes count: {len(known_set)}")

# Map Attack_type to integer labels: known classes 0..len(known_set)-1, withheld to len(known_set)
known_list = sorted(list(known_set))
label_to_int = {cls: idx for idx, cls in enumerate(known_list)}
unknown_class_index = len(known_list)  # if open_set else -1
print(f"Unknown class index (for withheld): {unknown_class_index}")

# Function to map a raw Attack_type string to integer label
def map_attack_type(val):
    if val in label_to_int:
        return label_to_int[val]
    else:
        # Withheld class
        return unknown_class_index

# Apply mapping to create integer label column
df['label_int'] = df['Attack_type'].apply(map_attack_type)

# Check mapping results
unknown_count = (df['label_int'] == unknown_class_index).sum()
known_counts = df['label_int'].value_counts()
print(f"Number of samples mapped to UNKNOWN (withheld): {unknown_count}")
print(f"Known label distribution (top 5): {known_counts.head()}")

# Now we need to split the data into train, val, test stratified by Attack_type (original string)
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
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
for train_idx, val_idx in sss2.split(X_train_val, y_train_val_str):
    pass

X_train = X_train_val[train_idx]
y_train_str = y_train_val_str[train_idx]
X_val = X_train_val[val_idx]
y_val_str = y_train_val_str[val_idx]

# Now we have indices relative to the original dataframe:
train_indices = train_val_idx[train_idx]
val_indices = train_val_idx[val_idx]
test_indices = test_idx

print(f"\nSplit sizes:")
print(f"  Training samples: {len(train_indices)}")
print(f"  Validation samples: {len(val_indices)}")
print(f"  Test samples: {len(test_indices)}")
print(f"  Percentages: train={len(train_indices)/len(df):.1%}, val={len(val_indices)/len(df):.1%}, test={len(test_indices)/len(df):.1%}")

# For open set, we need to ensure that withheld classes are excluded from training (but kept in val/test and mapped to unknown)
# We already have label_int where withheld classes are mapped to unknown_class_index.
# However we must also ensure that the scaler is fitted only on known training samples (i.e., those with label_int != unknown_class_index)
# So we will compute known-only mask for training.

known_train_mask = df.iloc[train_indices]['label_int'] != unknown_class_index
known_train_indices = train_indices[known_train_mask.values]
print(f"Known training samples (excluding withheld): {len(known_train_indices)}")
print(f"Withheld samples in training: {len(train_indices) - len(known_train_indices)}")

# Determine constant columns from training data only (after removing leakage columns, but before scaling)
# We'll use the training subset (original indices) and the feature columns.
# We'll consider only known training samples (label_int != unknown_class_index) if open_set, else all training samples.
train_known_features = df.iloc[known_train_indices][feature_cols]
constant_cols = []
for col in feature_cols:
    # Check if column has only one unique value (excluding NaN? but we have no NaN)
    if train_known_features[col].nunique() == 1:
        constant_cols.append(col)
if constant_cols:
    print(f"Constant columns found in training data (will be removed): {constant_cols}")
    print(f"Number of constant columns: {len(constant_cols)}")
    # Remove constant columns from feature columns
    feature_cols = [col for col in feature_cols if col not in constant_cols]
    # Update X arrays accordingly
    # We'll need to re-slice the data with the new feature columns
    X = df[feature_cols].values
    # Resplit indices? Instead we can recompute X_train, X_val, X_test using the new feature columns.
    X_train = df.iloc[train_indices][feature_cols].values
    X_val = df.iloc[val_indices][feature_cols].values
    X_test = df.iloc[test_indices][feature_cols].values
    y_train_str = df.iloc[train_indices]['Attack_type'].values
    y_val_str = df.iloc[val_indices]['Attack_type'].values
    y_test_str = df.iloc[test_indices]['Attack_type'].values
else:
    print("No constant columns found in training data.")

# After removing constant columns, we need to recompute the label_int mapping? No, label mapping unchanged.
# Now we have final feature columns.
final_input_dim = len(feature_cols)
print(f"\nFinal feature dimension (after leakage + constant removal): {final_input_dim}")
print(f"Retained feature names: {feature_cols}")

# Verify that no label/leakage columns remain in feature_cols
leakage_remaining = [c for c in feature_cols if c in label_cols or c in high_risk_cols or c in additional_cols]
if leakage_remaining:
    print(f"\nERROR: Leakage/label columns still present in features: {leakage_remaining}")
else:
    print(f"\nOK: No label/leakage columns remain in features.")

# For open set, we want to train only on known training samples (exclude withheld)
# Create mask for known training samples (label_int != unknown_class_index)
known_train_mask = df.iloc[train_indices]['label_int'] != unknown_class_index
known_train_indices = train_indices[known_train_mask.values]

# Build arrays for known-only training
X_train = df.iloc[known_train_indices][feature_cols].values
y_train = df.iloc[known_train_indices]['label_int'].values.astype(np.int64)

# Validation and test sets are the original splits (they include withheld samples mapped to unknown)
X_val = df.iloc[val_indices][feature_cols].values
y_val = df.iloc[val_indices]['label_int'].values.astype(np.int64)
X_test = df.iloc[test_indices][feature_cols].values
y_test = df.iloc[test_indices]['label_int'].values.astype(np.int64)

# Fit StandardScaler on known training data only
scaler = StandardScaler()
scaler.fit(X_train)
print(f"\nScaler fitted on {len(X_train)} known training samples.")
print(f"Scaler mean shape: {scaler.mean_.shape}")
print(f"Scaler scale shape: {scaler.scale_.shape}")

# Transform features
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check that scaler transforms correctly (should have zero mean, unit variance on training data)
print(f"Train scaled mean (approx): {np.mean(X_train_scaled, axis=0)[:5]}")
print(f"Train scaled std (approx): {np.std(X_train_scaled, axis=0)[:5]}")

print(f"\nLabel counts after mapping:")
print(f"  Training: unknown={(y_train==unknown_class_index).sum()}, known={len(y_train)- (y_train==unknown_class_index).sum()}")
print(f"  Validation: unknown={(y_val==unknown_class_index).sum()}, known={len(y_val)- (y_val==unknown_class_index).sum()}")
print(f"  Test: unknown={(y_test==unknown_class_index).sum()}, known={len(y_test)- (y_test==unknown_class_index).sum()}")

# Compute class weights (inverse frequency) for known classes only
# Only consider known classes in training (excluding unknown)
known_y_train = y_train[y_train != unknown_class_index]
if len(known_y_train) > 0:
    unique, counts = np.unique(known_y_train, return_counts=True)
    # Inverse frequency
    inv_freq = 1.0 / counts
    # Normalize so that sum = n_classes
    class_weights = inv_freq * len(unique) / inv_freq.sum()
    print(f"\nClass weights (known classes only, inverse frequency):")
    for idx, w in zip(unique, class_weights):
        class_name = known_list[idx] if idx < len(known_list) else f"unknown_{idx}"
        print(f"  {class_name} (idx {idx}): {w:.4f}")
else:
    print("\nNo known training samples (should not happen).")

print("\nSanity check completed successfully.")