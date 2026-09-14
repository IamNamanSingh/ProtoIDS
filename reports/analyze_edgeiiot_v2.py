"""
TASK 2 (completion): Edge-IIoT - remaining analysis items only.
We already know: 63 cols, 2,219,201 rows, Attack_label (binary), Attack_type (15 classes).
This script covers: dtypes, missing values, constant cols, identifier cols, correlations, stats.
"""
import pandas as pd
import numpy as np
import json
import os
import sys

BASE = r"c:\Users\LENOVO\Desktop\Capstone Project\DNN-EdgeIIoT-dataset.csv"
DATA_PATH = os.path.join(BASE, "DNN-EdgeIIoT-dataset.csv")
REPORT_DIR = r"c:\Users\LENOVO\Desktop\Capstone Project\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

results = {}

# Read head sample
print("Loading head sample (20000 rows)...")
sys.stdout.flush()
df_sample = pd.read_csv(DATA_PATH, nrows=20000, low_memory=False)
columns = list(df_sample.columns)
print(f"Shape: {df_sample.shape}")

# ============================================================
# Dtype analysis
# ============================================================
print("\n" + "=" * 70)
print("Dtype analysis")
print("=" * 70)

num_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"\nCategorical/object columns ({len(cat_cols)}): {cat_cols}")

results["num_cols"] = num_cols
results["cat_cols"] = cat_cols
sys.stdout.flush()

# ============================================================
# Missing values (chunked - only count, don't load full)
# ============================================================
print("\n" + "=" * 70)
print("Missing value analysis (chunked)")
print("=" * 70)
sys.stdout.flush()

missing_counts = pd.Series(0, index=columns, dtype='int64')
chunk_count = 0
for chunk in pd.read_csv(DATA_PATH, chunksize=200000, low_memory=False):
    missing_counts = missing_counts.add(chunk.isnull().sum(), fill_value=0).astype(int)
    chunk_count += 1
    if chunk_count % 5 == 0:
        print(f"  Processed {chunk_count} chunks ({chunk_count * 200000:,} rows)...")
        sys.stdout.flush()

print(f"Processed {chunk_count} chunks total")
total_missing = missing_counts.sum()
print(f"Total missing values: {total_missing:,}")
if total_missing > 0:
    print("\nMissing values per column:")
    for col in missing_counts[missing_counts > 0].sort_values(ascending=False).index:
        print(f"  {col}: {missing_counts[col]:,}")
else:
    print("No missing values found")

results["total_missing"] = int(total_missing)
results["missing_per_col"] = {str(k): int(v) for k, v in missing_counts[missing_counts > 0].items()}
sys.stdout.flush()

# ============================================================
# Duplicate rows (ESTIMATE from head sample)
# ============================================================
print("\n" + "=" * 70)
print("Duplicate rows (ESTIMATE from head sample)")
print("=" * 70)

dup_in_sample = df_sample.duplicated().sum()
dup_pct = 100 * dup_in_sample / len(df_sample)
print(f"Duplicates in head sample (20000 rows): {dup_in_sample:,} ({dup_pct:.2f}%)")
print("NOTE: ESTIMATE only from first 20K rows.")
sys.stdout.flush()

# ============================================================
# Constant / near-constant columns
# ============================================================
print("\n" + "=" * 70)
print("Constant / near-constant columns")
print("=" * 70)

for col in df_sample.columns:
    nunique = df_sample[col].nunique(dropna=False)
    if nunique <= 1:
        print(f"  CONSTANT: '{col}' = {df_sample[col].unique()[:3]}")
    elif nunique <= 5:
        vc = df_sample[col].value_counts(normalize=True)
        top_pct = vc.iloc[0] * 100
        if top_pct >= 95.0:
            print(f"  NEAR-CONSTANT: '{col}' -- top={top_pct:.1f}% ({nunique} unique)")
sys.stdout.flush()

# ============================================================
# Identifier columns
# ============================================================
print("\n" + "=" * 70)
print("Identifier columns")
print("=" * 70)

identifier_keywords = ["ip", "addr", "port", "time", "stamp", "id", "flow", "src", "dst",
                        "source", "dest", "mac", "host", "url", "file", "path", "name"]

for col in columns:
    col_lower = col.lower()
    for kw in identifier_keywords:
        if kw in col_lower:
            sample_vals = df_sample[col].dropna().head(3).tolist()
            print(f"  '{col}' (keyword: '{kw}') | dtype: {df_sample[col].dtype} | unique: {df_sample[col].nunique()} | sample: {sample_vals}")
            break
sys.stdout.flush()

# ============================================================
# Correlation with Attack_type (leakage)
# ============================================================
print("\n" + "=" * 70)
print("Feature-label correlation (Attack_type)")
print("=" * 70)

main_label = 'Attack_type'
if main_label in df_sample.columns:
    label_encoded, _ = pd.factorize(df_sample[main_label])
    correlations = {}
    for col in num_cols:
        if col in ['Attack_label']:
            continue
        try:
            vals = df_sample[col].fillna(0).replace([np.inf, -np.inf], 0).values.astype(float)
            corr = np.corrcoef(vals, label_encoded)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            pass

    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 features most correlated with Attack_type (from head sample):")
    for col, corr in sorted_corr[:20]:
        flag = " *** SUSPICIOUS" if corr > 0.8 else (" ** HIGH" if corr > 0.5 else "")
        print(f"  {col}: {corr:.4f}{flag}")

    results["top_correlations"] = [(c, float(v)) for c, v in sorted_corr[:20]]
sys.stdout.flush()

# ============================================================
# Basic statistics of numerical features
# ============================================================
print("\n" + "=" * 70)
print("Basic statistics (from head sample)")
print("=" * 70)

if num_cols:
    desc = df_sample[num_cols].describe().T
    print(desc[['mean', 'std', 'min', 'max']].to_string())
sys.stdout.flush()

# ============================================================
# Cross-tab Attack_label vs Attack_type
# ============================================================
print("\n" + "=" * 70)
print("Cross-tab: Attack_label vs Attack_type")
print("=" * 70)

if 'Attack_label' in df_sample.columns and 'Attack_type' in df_sample.columns:
    ct = pd.crosstab(df_sample['Attack_type'], df_sample['Attack_label'])
    print(ct.to_string())
sys.stdout.flush()

# Save results
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

print("\nResults saved to reports/edgeiiot_analysis.json")
print("\nEDGE-IIoTset ANALYSIS COMPLETE")
sys.stdout.flush()
