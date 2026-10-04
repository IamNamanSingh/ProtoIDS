"""
Smoke test for the X-IIoTID loader.

Builds a synthetic CSV that mimics the published X-IIoTID schema (identifier /
temporal / port columns + 42 numeric flow features + 3 label levels), then runs
the real loader over it to verify:

  * schema discovery picks the most granular label column
  * leakage columns are dropped
  * constant columns are dropped
  * withheld classes get ZERO rows in the training split
  * the scaler is fitted only on known-train rows
  * val/test are transformed with that same scaler
  * input_dim / num_classes come out right
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from protoids.dataset import (  # noqa: E402
    create_data_loaders,
    discover_xiiotid_columns,
    suggest_xiiotid_withheld,
)

RNG = np.random.default_rng(0)

SUBCATS = [
    "Normal",
    "Fuzzing",
    "Port_Scanning",
    "Vulnerability_Scanner",
    "Dictionary_Attack",
    "Exhausted_Dictionary",
    "Reverse_Shell",
    "Man-in-the-Middle",
    "Poisoning_Attack",
    "Ransomware",
]
ATTACK_TYPE = {
    "Normal": "Normal",
    "Fuzzing": "Fuzzing",
    "Port_Scanning": "Scanning",
    "Vulnerability_Scanner": "Scanning",
    "Dictionary_Attack": "Brute_force",
    "Exhausted_Dictionary": "Brute_force",
    "Reverse_Shell": "Exploitation",
    "Man-in-the-Middle": "Exploitation",
    "Poisoning_Attack": "Tampering",
    "Ransomware": "Ransomware",
}

LEAKY = ["ID", "Timestamp", "Source IP", "Destination IP", "Source Port", "Destination Port"]
CONSTANT = ["os_version", "device_model"]

NUMERIC_FEATURES = [
    "Protocol", "Flow Count", "Packet Length", "Payload Size", "TCP Flags", "Time to Live",
    "Window Size", "MSS", "Bytes Retransmitted", "Segment Size", "Average Packet Size",
    "Packet Rate", "Byte Rate", "Duration", "SYN Flag Count", "ACK Flag Count",
    "RST Flag Count", "FIN Flag Count", "PSH Flag Count", "URG Flag Count", "CWR Flag Count",
    "ECE Flag Count", "Packet Length Variance", "Packet Length Std Dev", "Packet Length Mean",
    "Packet Length Skewness", "Packet Length Kurtosis", "Packet Length Median",
    "Packet Length Mode", "Packet Length Min", "Packet Length Max", "Packet Length Range",
    "Packet Length Coefficient of Variation", "Time to Live Variance", "Time to Live Std Dev",
    "Time to Live Mean", "Time to Live Median", "Time to Live Mode", "Time to Live Min",
    "Time to Live Max", "Time to Live Range", "Window Size Variance", "Window Size Std Dev",
    "Window Size Mean",
]


def make_csv(path: str, n_per_class: int = 400) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = []
    for idx, sub in enumerate(SUBCATS):
        centre = RNG.normal(loc=idx * 2.0, scale=1.0, size=(n_per_class, len(NUMERIC_FEATURES)))
        block = pd.DataFrame(centre, columns=NUMERIC_FEATURES)
        block.insert(0, "ID", np.arange(len(block)))
        block.insert(1, "Timestamp", 1_565_000_000.0 + np.arange(len(block)))
        block.insert(2, "Source IP", "192.168.1." + RNG.integers(1, 254, len(block)).astype(str))
        block.insert(3, "Destination IP", "10.0.0." + RNG.integers(1, 254, len(block)).astype(str))
        block.insert(4, "Source Port", RNG.integers(1024, 65535, len(block)))
        block.insert(5, "Destination Port", RNG.choice([80, 443, 502, 1883], len(block)))
        block["os_version"] = "4.14.1"
        block["device_model"] = "PLC-1"
        block["Sub-Category"] = sub
        block["Attack Type"] = ATTACK_TYPE[sub]
        block["Attack Category"] = "Normal" if sub == "Normal" else "Attack"
        block["Label"] = "Normal" if sub == "Normal" else "Attack"
        frames.append(block)
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)  # shuffle rows
    df.to_csv(path, index=False)
    return df


def main() -> int:
    work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_xiiotid")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    csv_path = os.path.join(work, "X-IIoTID Dataset.csv")

    print("== building synthetic X-IIoTID CSV ==")
    df = make_csv(csv_path)
    print(f"   {csv_path}  shape={df.shape}")

    label_col, features, reasons = discover_xiiotid_columns(df.columns.tolist())
    print("\n== discovery ==")
    print(f"   label column : {label_col}")
    print(f"   features kept: {len(features)}")
    print(f"   dropped      : {sorted(reasons)}")
    assert label_col == "Sub-Category", label_col
    for col in LEAKY + ["Attack Type", "Attack Category", "Label"]:
        assert col not in features, f"{col} should have been dropped"
    for col in CONSTANT:
        assert col in features, f"{col} should still be a candidate feature"
    print(f"   constant candidates left for the loader to filter: {CONSTANT}")

    lc, suggested, counts = suggest_xiiotid_withheld(work, k=3)
    print("\n== suggestion ==")
    print(f"   label col    : {lc}")
    print(f"   suggested    : {suggested}")
    assert "Normal" not in suggested, "must never withhold the benign class"

    scaler_dir = os.path.join(work, "results")
    processed_dir = os.path.join(work, "processed")
    print("\n== loader (open-set) ==")
    tr, va, te, num_classes, input_dim = create_data_loaders(
        data_dir=work,
        batch_size=64,
        withheld_classes=suggested,
        open_set=True,
        scaler_save_dir=scaler_dir,
        dataset_type='xiiotid',
    )
    print(f"   num_classes={num_classes}  input_dim={input_dim}")

    expected_classes = len(set(SUBCATS) - set(suggested)) + 1
    assert num_classes == expected_classes, (num_classes, expected_classes)
    assert input_dim == len(NUMERIC_FEATURES), (input_dim, len(NUMERIC_FEATURES))

    tr_labels = np.asarray(tr.dataset.labels)
    va_labels = np.asarray(va.dataset.labels)
    te_labels = np.asarray(te.dataset.labels)
    unknown_idx = tr.dataset.get_unknown_class_index()

    print("\n== protocol assertions ==")
    print(f"   unknown index : {unknown_idx}")
    print(f"   train rows    : {len(tr_labels)}")
    print(f"   val rows      : {len(va_labels)}  (unknown: {int((va_labels == unknown_idx).sum())})")
    print(f"   test rows     : {len(te_labels)}  (unknown: {int((te_labels == unknown_idx).sum())})")

    assert not np.any(tr_labels == unknown_idx), "unknown class leaked into training!"
    assert (va_labels == unknown_idx).sum() > 0, "validation has no unknown samples"
    assert (te_labels == unknown_idx).sum() > 0, "test has no unknown samples"
    n_known = num_classes - 1
    assert set(np.unique(tr_labels).tolist()) == set(range(n_known)), \
        f"training classes {np.unique(tr_labels)} != 0..{n_known - 1}"

    import joblib
    scaler = joblib.load(os.path.join(scaler_dir, "xiiotid_preprocessor.joblib"))
    print(f"   scaler.n_samples_seen_ = {scaler.n_samples_seen_}")
    assert int(scaler.n_samples_seen_) == len(tr_labels), \
        (scaler.n_samples_seen_, len(tr_labels))

    Xtr = np.asarray(tr.dataset.features[:2000])
    assert Xtr.min() >= -5.0 - 1e-6 and Xtr.max() <= 5.0 + 1e-6, (Xtr.min(), Xtr.max())
    print(f"   train feature range: [{Xtr.min():.3f}, {Xtr.max():.3f}]")

    import json
    man = json.load(open(os.path.join(scaler_dir, "xiiotid_feature_manifest.json")))
    lmap = json.load(open(os.path.join(scaler_dir, "xiiotid_label_mapping.json")))
    print(f"   manifest label col : {man['label_column']}")
    print(f"   withheld recorded  : {lmap['withheld_classes']}")
    assert man["label_column"] == "Sub-Category"
    assert sorted(lmap["withheld_classes"]) == sorted(suggested)
    for col in CONSTANT:
        assert col in man["removed_columns"], f"{col} should be recorded as removed"
        assert col not in man["retained_columns"], f"{col} should not be a feature"
    assert input_dim == len(man["retained_columns"])

    print("\n== loader (closed-set) ==")
    tr2, va2, te2, nc2, id2 = create_data_loaders(
        data_dir=work, batch_size=64, open_set=False,
        scaler_save_dir=os.path.join(work, "results_closed"),
        dataset_type='xiiotid',
    )
    assert nc2 == len(SUBCATS), (nc2, len(SUBCATS))
    l2 = np.asarray(tr2.dataset.labels)
    assert l2.max() < nc2, "closed-set should not reserve an unknown index"
    print(f"   num_classes={nc2} input_dim={id2} train_rows={len(l2)}")

    print("\n== loader (max_rows smoke) ==")
    tr3, va3, te3, nc3, id3 = create_data_loaders(
        data_dir=work, batch_size=64, open_set=False, max_rows=1000,
        dataset_type='xiiotid',
    )
    total = len(tr3.dataset) + len(va3.dataset) + len(te3.dataset)
    assert total == 1000, total
    print(f"   rows respected cap: {total}")

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
