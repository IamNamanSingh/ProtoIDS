"""
TASK 1: Forensic analysis of CICIoT2023 dataset
Memory-efficient analysis of train, validation, and test splits.
"""
import pandas as pd
import numpy as np
import json
import os
import sys

BASE = r"c:\Users\LENOVO\Desktop\Capstone Project\CICIOT23"
REPORT_DIR = r"c:\Users\LENOVO\Desktop\Capstone Project\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

results = {}

# ============================================================
# STEP 1: Read headers from all three files first
# ============================================================
print("=" * 70)
print("STEP 1: Reading headers")
print("=" * 70)

splits = {
    "train": os.path.join(BASE, "train", "train.csv"),
    "validation": os.path.join(BASE, "validation", "validation.csv"),
    "test": os.path.join(BASE, "test", "test.csv"),
}

headers = {}
for name, path in splits.items():
    df_head = pd.read_csv(path, nrows=0)
    headers[name] = list(df_head.columns)
    print(f"\n{name}: {len(headers[name])} columns")
    print(f"  Columns: {headers[name]}")

# Check if all splits have the same columns
print("\n--- Column Consistency Check ---")
train_cols = set(headers["train"])
val_cols = set(headers["validation"])
test_cols = set(headers["test"])
print(f"Train columns == Validation columns: {train_cols == val_cols}")
print(f"Train columns == Test columns: {train_cols == test_cols}")
if train_cols != val_cols:
    print(f"  In train but not validation: {train_cols - val_cols}")
    print(f"  In validation but not train: {val_cols - train_cols}")
if train_cols != test_cols:
    print(f"  In train but not test: {train_cols - test_cols}")
    print(f"  In test but not train: {test_cols - train_cols}")

results["column_names"] = headers["train"]
results["columns_consistent"] = (train_cols == val_cols == test_cols)

# ============================================================
# STEP 2: Count rows efficiently (without loading full data)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Counting rows")
print("=" * 70)

row_counts = {}
for name, path in splits.items():
    count = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for _ in f:
            count += 1
    count -= 1  # subtract header
    row_counts[name] = count
    print(f"{name}: {count:,} rows")

results["row_counts"] = row_counts

# ============================================================
# STEP 3: Load data (chunked for train, full for val/test)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Loading data for detailed analysis")
print("=" * 70)

# Load validation and test fully (they're ~332MB each, manageable)
print("Loading validation...")
df_val = pd.read_csv(splits["validation"], low_memory=False)
print(f"  Shape: {df_val.shape}")

print("Loading test...")
df_test = pd.read_csv(splits["test"], low_memory=False)
print(f"  Shape: {df_test.shape}")

# For train, load in chunks to analyze
print("Loading train (chunked)...")
try:
    df_train = pd.read_csv(splits["train"], low_memory=False)
    print(f"  Shape: {df_train.shape}")
    train_loaded_fully = True
except MemoryError:
    print("  Memory error loading train fully, using chunked approach")
    train_loaded_fully = False

# ============================================================
# STEP 4: Identify label columns
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Identifying label columns")
print("=" * 70)

# Check for columns that look like labels
label_candidates = []
for col in headers["train"]:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ["label", "class", "attack", "category", "type", "target"]):
        label_candidates.append(col)

print(f"Label candidate columns: {label_candidates}")

# If no obvious candidates, check the last few columns and all object columns
if not label_candidates:
    print("No obvious label columns found by keyword. Checking all object columns and last columns...")
    ref_check = df_val
    obj_cols = ref_check.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        if col not in label_candidates:
            label_candidates.append(col)
    for col in headers["train"][-3:]:
        if col not in label_candidates:
            label_candidates.append(col)
    print(f"Expanded candidates: {label_candidates}")

# Check dtypes and unique values of candidates
ref_df = df_val  # use validation for quick checks
for col in label_candidates:
    if col in ref_df.columns:
        print(f"\n  Column: '{col}'")
        print(f"  Dtype: {ref_df[col].dtype}")
        nunique = ref_df[col].nunique()
        print(f"  Unique values: {nunique}")
        if nunique <= 50:
            print(f"  Values: {sorted(ref_df[col].astype(str).unique().tolist())}")

results["label_candidates"] = label_candidates

# ============================================================
# STEP 5: Detailed label analysis on all splits
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Label analysis across splits")
print("=" * 70)

for name, df in [("train", df_train if train_loaded_fully else None),
                 ("validation", df_val), ("test", df_test)]:
    if df is None:
        continue
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n{name} - Object/string columns: {obj_cols}")
    for col in obj_cols:
        nunique = df[col].nunique()
        print(f"  '{col}': {nunique} unique values")
        if nunique <= 60:
            vc = df[col].value_counts()
            print(f"  Distribution:\n{vc.to_string()}")

# ============================================================
# STEP 6: Dtype analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Dtype analysis")
print("=" * 70)

ref_df2 = df_val
dtypes = ref_df2.dtypes
num_cols = ref_df2.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = ref_df2.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = ref_df2.select_dtypes(include=['bool']).columns.tolist()

print(f"Numerical columns: {len(num_cols)}")
print(f"Categorical/object columns: {len(cat_cols)}")
print(f"Boolean columns: {len(bool_cols)}")
print(f"\nNumerical: {num_cols}")
print(f"\nCategorical: {cat_cols}")

results["num_columns"] = len(num_cols)
results["cat_columns"] = len(cat_cols)
results["num_col_names"] = num_cols
results["cat_col_names"] = cat_cols

# ============================================================
# STEP 7: Missing values
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Missing value analysis")
print("=" * 70)

for name, df in [("train", df_train if train_loaded_fully else None),
                 ("validation", df_val), ("test", df_test)]:
    if df is None:
        continue
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"\n{name}: Total missing values = {total_missing}")
    if total_missing > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values")

# Also check for inf values in numerical columns
print("\n--- Checking for inf values ---")
for name, df in [("train", df_train if train_loaded_fully else None),
                 ("validation", df_val), ("test", df_test)]:
    if df is None:
        continue
    num_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(num_df.values).sum()
    print(f"{name}: inf values = {inf_count}")
    if inf_count > 0:
        for col in num_df.columns:
            c = np.isinf(df[col].values).sum()
            if c > 0:
                print(f"  {col}: {c} inf values")

# ============================================================
# STEP 8: Duplicate rows
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Duplicate rows")
print("=" * 70)

for name, df in [("validation", df_val), ("test", df_test)]:
    dup_count = df.duplicated().sum()
    print(f"{name}: {dup_count:,} duplicate rows ({100*dup_count/len(df):.2f}%)")

if train_loaded_fully:
    dup_count_train = df_train.duplicated().sum()
    print(f"train: {dup_count_train:,} duplicate rows ({100*dup_count_train/len(df_train):.2f}%)")

# ============================================================
# STEP 9: Constant / near-constant columns
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Constant / near-constant columns")
print("=" * 70)

ref = df_val
print("Using validation set for constant/near-constant check:")
for col in ref.columns:
    nunique = ref[col].nunique(dropna=False)
    if nunique <= 1:
        print(f"  CONSTANT: '{col}' has {nunique} unique value(s): {ref[col].unique()[:5]}")
    elif nunique <= 5:
        vc = ref[col].value_counts(normalize=True)
        top_pct = vc.iloc[0] * 100
        if top_pct >= 99.0:
            print(f"  NEAR-CONSTANT: '{col}' -- top value covers {top_pct:.2f}% ({nunique} unique)")

if train_loaded_fully:
    print("\nUsing train set for constant/near-constant check:")
    for col in df_train.columns:
        nunique = df_train[col].nunique(dropna=False)
        if nunique <= 1:
            print(f"  CONSTANT: '{col}' has {nunique} unique value(s): {df_train[col].unique()[:5]}")
        elif nunique <= 5:
            vc = df_train[col].value_counts(normalize=True)
            top_pct = vc.iloc[0] * 100
            if top_pct >= 99.0:
                print(f"  NEAR-CONSTANT: '{col}' -- top value covers {top_pct:.2f}% ({nunique} unique)")

# ============================================================
# STEP 10: IP / Timestamp / Port / ID columns
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Identifier columns (IP, timestamp, port, etc.)")
print("=" * 70)

identifier_keywords = ["ip", "addr", "port", "time", "stamp", "id", "flow", "src", "dst",
                        "source", "dest", "mac", "host", "url", "file", "path", "name"]

for col in headers["train"]:
    col_lower = col.lower()
    for kw in identifier_keywords:
        if kw in col_lower:
            print(f"  POTENTIAL IDENTIFIER: '{col}' (matched keyword: '{kw}')")
            if col in ref_df2.columns:
                sample_vals = ref_df2[col].dropna().head(5).tolist()
                print(f"    Sample values: {sample_vals}")
                print(f"    Dtype: {ref_df2[col].dtype}, Unique: {ref_df2[col].nunique()}")
            break

# ============================================================
# STEP 11: Correlation with label (leakage detection)
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Feature-label correlation (leakage detection)")
print("=" * 70)

main_label = None
for col in label_candidates:
    if col in ref_df2.columns and ref_df2[col].dtype == 'object':
        main_label = col
        break

if main_label is None and label_candidates:
    main_label = label_candidates[-1]

print(f"Main label column identified: '{main_label}'")

if main_label and main_label in ref_df2.columns:
    label_encoded, _ = pd.factorize(ref_df2[main_label])
    correlations = {}
    for col in num_cols:
        try:
            vals = ref_df2[col].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
            corr = np.corrcoef(vals, label_encoded)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            pass

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 features most correlated with label:")
    for col, corr in sorted_corr[:20]:
        flag = " *** SUSPICIOUS" if corr > 0.8 else (" ** HIGH" if corr > 0.5 else "")
        print(f"  {col}: {corr:.4f}{flag}")

    results["top_correlations"] = [(c, float(v)) for c, v in sorted_corr[:20]]

# ============================================================
# STEP 12: Class distribution comparison across splits
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Class distribution comparison")
print("=" * 70)

if main_label:
    for name, df in [("train", df_train if train_loaded_fully else None),
                     ("validation", df_val), ("test", df_test)]:
        if df is None:
            continue
        if main_label in df.columns:
            print(f"\n--- {name} ---")
            vc = df[main_label].value_counts()
            vc_pct = df[main_label].value_counts(normalize=True) * 100
            combined = pd.DataFrame({"count": vc, "pct": vc_pct})
            print(combined.to_string())
            print(f"Total classes: {df[main_label].nunique()}")

# ============================================================
# STEP 13: Check for multiple label hierarchies
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: Label hierarchy check")
print("=" * 70)

if len(label_candidates) >= 2:
    print(f"Multiple label columns found: {label_candidates}")
    for col in label_candidates:
        if col in ref_df2.columns:
            print(f"\n  '{col}': {ref_df2[col].nunique()} unique values")
            if ref_df2[col].nunique() <= 50:
                print(f"  Values: {sorted(ref_df2[col].astype(str).unique().tolist())}")
elif len(label_candidates) == 1:
    lbl_col = label_candidates[0]
    if lbl_col in ref_df2.columns:
        vals = ref_df2[lbl_col].unique()
        print(f"Single label column '{lbl_col}' with {len(vals)} values")
        if ref_df2[lbl_col].dtype == 'object':
            prefixes = set()
            for v in vals:
                if '-' in str(v):
                    prefixes.add(str(v).split('-')[0])
                elif '_' in str(v):
                    prefixes.add(str(v).split('_')[0])
            if prefixes:
                print(f"  Potential categories from prefixes: {sorted(prefixes)}")

# ============================================================
# STEP 14: Basic statistics of numerical features
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: Basic statistics (validation set)")
print("=" * 70)

desc = ref_df2[num_cols].describe().T
print(desc[['mean', 'std', 'min', 'max']].to_string())

# ============================================================
# STEP 15: Save intermediate results
# ============================================================
print("\n" + "=" * 70)
print("STEP 15: Saving results")
print("=" * 70)

results["main_label"] = main_label

def make_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

serializable_results = {}
for k, v in results.items():
    if isinstance(v, dict):
        serializable_results[k] = {str(kk): make_serializable(vv) for kk, vv in v.items()}
    elif isinstance(v, list):
        serializable_results[k] = [make_serializable(i) if not isinstance(i, tuple) else [make_serializable(x) for x in i] for i in v]
    else:
        serializable_results[k] = make_serializable(v)

with open(os.path.join(REPORT_DIR, "ciciot23_analysis.json"), 'w') as f:
    json.dump(serializable_results, f, indent=2, default=str)

print("Results saved to reports/ciciot23_analysis.json")
print("\nCICIoT2023 ANALYSIS COMPLETE")
