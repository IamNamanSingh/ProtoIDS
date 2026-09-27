"""
TASK 2: Forensic analysis of Edge-IIoTset DNN dataset
FAST approach: head sample + chunked aggregations only.
No skiprows random sampling (too slow for 1.2 GB).
"""
import pandas as pd
import numpy as np
import json
import os
import gc
import sys

BASE = r"c:\Users\LENOVO\Desktop\Capstone Project\DNN-EdgeIIoT-dataset.csv"
DATA_PATH = os.path.join(BASE, "DNN-EdgeIIoT-dataset.csv")
REPORT_DIR = r"c:\Users\LENOVO\Desktop\Capstone Project\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

results = {}

# ============================================================
# STEP 1: Read header only
# ============================================================
print("=" * 70)
print("STEP 1: Reading headers")
print("=" * 70)
sys.stdout.flush()

df_head = pd.read_csv(DATA_PATH, nrows=0)
columns = list(df_head.columns)
print(f"Number of columns: {len(columns)}")
print(f"Columns: {columns}")
results["columns"] = columns
results["num_columns_total"] = len(columns)
sys.stdout.flush()

# ============================================================
# STEP 2: Count rows efficiently
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Counting rows")
print("=" * 70)
sys.stdout.flush()

row_count = 0
with open(DATA_PATH, 'r', encoding='utf-8', errors='replace') as f:
    for _ in f:
        row_count += 1
row_count -= 1  # subtract header
print(f"Total rows: {row_count:,}")
results["row_count"] = row_count
sys.stdout.flush()

# ============================================================
# STEP 3: Read first 20000 rows for dtype/structure analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Reading head sample (20000 rows)")
print("=" * 70)
sys.stdout.flush()

df_sample = pd.read_csv(DATA_PATH, nrows=20000, low_memory=False)
print(f"Sample shape: {df_sample.shape}")
sys.stdout.flush()

# ============================================================
# STEP 4: Identify label columns
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Identifying label columns")
print("=" * 70)
sys.stdout.flush()

label_candidates = []
for col in columns:
    col_lower = col.lower()
    if any(kw in col_lower for kw in ["label", "class", "attack", "category", "type", "target"]):
        label_candidates.append(col)

# Also check object columns
obj_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
for col in obj_cols:
    if col not in label_candidates:
        label_candidates.append(col)

print(f"Label candidate columns: {label_candidates}")

for col in label_candidates:
    if col in df_sample.columns:
        print(f"\n  Column: '{col}'")
        print(f"  Dtype: {df_sample[col].dtype}")
        nunique = df_sample[col].nunique()
        print(f"  Unique values in sample: {nunique}")
        if nunique <= 50:
            print(f"  Values: {sorted(df_sample[col].astype(str).unique().tolist())}")
        else:
            print(f"  (Too many unique values to list, showing first 10 samples)")
            print(f"  Sample: {df_sample[col].dropna().head(10).tolist()}")

results["label_candidates"] = label_candidates
sys.stdout.flush()

# ============================================================
# STEP 5: Full label distribution (chunked) - ONLY for label cols with <=50 unique
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Full label distribution (chunked)")
print("=" * 70)
sys.stdout.flush()

# Determine which columns to get full distribution for
label_cols_for_dist = []
for col in label_candidates:
    if col in df_sample.columns and df_sample[col].nunique() <= 50:
        label_cols_for_dist.append(col)

if not label_cols_for_dist:
    # Use the last column as likely label
    label_cols_for_dist = [columns[-1]]

print(f"Getting full distribution for: {label_cols_for_dist}")

for lbl_col in label_cols_for_dist:
    print(f"\nLabel column: '{lbl_col}'")
    sys.stdout.flush()
    label_counts = pd.Series(dtype='int64')
    chunk_size = 200000
    chunk_num = 0
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size, usecols=[lbl_col], low_memory=False):
        vc = chunk[lbl_col].value_counts()
        label_counts = label_counts.add(vc, fill_value=0).astype(int)
        chunk_num += 1
        if chunk_num % 5 == 0:
            print(f"  Processed {chunk_num * chunk_size:,} rows...")
            sys.stdout.flush()

    label_counts = label_counts.sort_values(ascending=False)
    total = label_counts.sum()
    print(f"  Total rows: {total:,}")
    print(f"  Unique values: {len(label_counts)}")
    print(f"  Distribution:")
    for val, count in label_counts.items():
        print(f"    {val}: {count:,} ({100*count/total:.2f}%)")

    results[f"label_dist_{lbl_col}"] = {str(k): int(v) for k, v in label_counts.items()}
    sys.stdout.flush()

# ============================================================
# STEP 6: Dtype analysis (from head sample)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Dtype analysis")
print("=" * 70)

num_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
bool_cols = df_sample.select_dtypes(include=['bool']).columns.tolist()

print(f"Numerical columns: {len(num_cols)}")
print(f"Categorical/object columns: {len(cat_cols)}")
print(f"Boolean columns: {len(bool_cols)}")
print(f"\nNumerical: {num_cols}")
print(f"\nCategorical: {cat_cols}")

results["num_col_count"] = len(num_cols)
results["cat_col_count"] = len(cat_cols)
results["num_col_names"] = num_cols
results["cat_col_names"] = cat_cols
sys.stdout.flush()

# ============================================================
# STEP 7: Missing values (chunked)
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Missing value analysis (chunked)")
print("=" * 70)
sys.stdout.flush()

missing_counts = pd.Series(0, index=columns, dtype='int64')
chunk_count = 0
for chunk in pd.read_csv(DATA_PATH, chunksize=200000, low_memory=False):
    missing_counts = missing_counts.add(chunk.isnull().sum(), fill_value=0).astype(int)
    chunk_count += 1
    if chunk_count % 5 == 0:
        print(f"  Processed {chunk_count} chunks...")
        sys.stdout.flush()

print(f"Processed {chunk_count} chunks total")
total_missing = missing_counts.sum()
print(f"Total missing values: {total_missing}")
if total_missing > 0:
    print("\nMissing values per column:")
    for col in missing_counts[missing_counts > 0].index:
        print(f"  {col}: {missing_counts[col]:,}")
else:
    print("No missing values found")

results["total_missing"] = int(total_missing)
results["missing_per_col"] = {str(k): int(v) for k, v in missing_counts[missing_counts > 0].items()}
sys.stdout.flush()

# ============================================================
# STEP 8: Duplicate rows (ESTIMATE from head sample)
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Duplicate rows (ESTIMATE from head sample)")
print("=" * 70)

dup_in_sample = df_sample.duplicated().sum()
dup_pct_sample = 100 * dup_in_sample / len(df_sample)
estimated_dups = int(dup_pct_sample / 100 * row_count)
print(f"Duplicates in head sample ({len(df_sample):,} rows): {dup_in_sample:,} ({dup_pct_sample:.2f}%)")
print(f"ESTIMATED duplicates in full dataset: ~{estimated_dups:,}")
print("NOTE: This is an ESTIMATE from the first 20K rows. Actual count may differ significantly.")

results["dup_sample_count"] = int(dup_in_sample)
results["dup_sample_pct"] = float(dup_pct_sample)
results["dup_estimated_full"] = estimated_dups
sys.stdout.flush()

# ============================================================
# STEP 9: Constant / near-constant columns (from head sample)
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Constant / near-constant columns (from head sample)")
print("=" * 70)

const_cols = []
near_const_cols = []
for col in df_sample.columns:
    nunique = df_sample[col].nunique(dropna=False)
    if nunique <= 1:
        print(f"  CONSTANT: '{col}' has {nunique} unique value(s): {df_sample[col].unique()[:5]}")
        const_cols.append(col)
    elif nunique <= 5:
        vc = df_sample[col].value_counts(normalize=True)
        top_pct = vc.iloc[0] * 100
        if top_pct >= 99.0:
            print(f"  NEAR-CONSTANT: '{col}' -- top value covers {top_pct:.2f}% ({nunique} unique)")
            near_const_cols.append(col)

results["constant_columns"] = const_cols
results["near_constant_columns"] = near_const_cols
sys.stdout.flush()

# ============================================================
# STEP 10: IP / Timestamp / Port / ID columns
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Identifier columns (IP, timestamp, port, etc.)")
print("=" * 70)

identifier_keywords = ["ip", "addr", "port", "time", "stamp", "id", "flow", "src", "dst",
                        "source", "dest", "mac", "host", "url", "file", "path", "name"]

identifier_cols_found = []
for col in columns:
    col_lower = col.lower()
    for kw in identifier_keywords:
        if kw in col_lower:
            print(f"  POTENTIAL IDENTIFIER: '{col}' (matched keyword: '{kw}')")
            if col in df_sample.columns:
                sample_vals = df_sample[col].dropna().head(5).tolist()
                print(f"    Sample values: {sample_vals}")
                print(f"    Dtype: {df_sample[col].dtype}, Unique in sample: {df_sample[col].nunique()}")
            identifier_cols_found.append(col)
            break

results["identifier_columns"] = identifier_cols_found
sys.stdout.flush()

# ============================================================
# STEP 11: Leakage analysis (correlation with label from head sample)
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Leakage analysis")
print("=" * 70)

# Determine main label
main_label = None
for col in label_cols_for_dist:
    if col in df_sample.columns:
        if df_sample[col].dtype == 'object' or df_sample[col].nunique() <= 30:
            main_label = col
            break

if main_label is None:
    # Fallback: use last column
    main_label = columns[-1]

print(f"Main label: '{main_label}'")

if main_label and main_label in df_sample.columns:
    label_encoded, _ = pd.factorize(df_sample[main_label])
    correlations = {}
    for col in num_cols:
        if col == main_label:
            continue
        try:
            vals = df_sample[col].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
            corr = np.corrcoef(vals, label_encoded)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            pass

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 features most correlated with label (from head sample):")
    for col, corr in sorted_corr[:20]:
        flag = " *** SUSPICIOUS" if corr > 0.8 else (" ** HIGH" if corr > 0.5 else "")
        print(f"  {col}: {corr:.4f}{flag}")

    results["top_correlations"] = [(c, float(v)) for c, v in sorted_corr[:20]]
sys.stdout.flush()

# ============================================================
# STEP 12: Label hierarchy check
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Label hierarchy check")
print("=" * 70)

if len(label_candidates) >= 2:
    print(f"Multiple label columns: {label_candidates}")
    for col in label_candidates:
        if col in df_sample.columns:
            nunique = df_sample[col].nunique()
            print(f"\n  '{col}': {nunique} unique values in sample")
            if nunique <= 30:
                vals = sorted(df_sample[col].astype(str).unique().tolist())
                print(f"  Values: {vals}")

    # Check cross-tabulation between first two label columns with <=30 unique
    small_label_cols = [c for c in label_candidates if c in df_sample.columns and df_sample[c].nunique() <= 30]
    if len(small_label_cols) >= 2:
        col1, col2 = small_label_cols[0], small_label_cols[1]
        print(f"\nCross-tab '{col1}' vs '{col2}':")
        ct = pd.crosstab(df_sample[col1], df_sample[col2])
        print(ct.to_string())
sys.stdout.flush()

# ============================================================
# STEP 13: Basic statistics
# ============================================================
print("\n" + "=" * 70)
print("STEP 13: Basic statistics (from head sample)")
print("=" * 70)

if num_cols:
    desc = df_sample[num_cols].describe().T
    print(desc[['mean', 'std', 'min', 'max']].to_string())
sys.stdout.flush()

# ============================================================
# STEP 14: Save results
# ============================================================
print("\n" + "=" * 70)
print("STEP 14: Saving results")
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

with open(os.path.join(REPORT_DIR, "edgeiiot_analysis.json"), 'w') as f:
    json.dump(serializable_results, f, indent=2, default=str)

print("Results saved to reports/edgeiiot_analysis.json")
print("\nEDGE-IIoTset ANALYSIS COMPLETE")
sys.stdout.flush()
