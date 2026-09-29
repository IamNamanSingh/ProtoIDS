#!/usr/bin/env python3
"""
Standalone script to analyze ProtoIDS threshold operating point.

Loads saved model and validation data, sweeps threshold, computes metrics,
and analyzes the distribution of minimum prototype distances.

DOES NOT modify any source files or experiment results.
"""

import argparse
import os
import sys
import json
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


# Add src/ to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from protoids.dataset import create_data_loaders
from protoids.protoids_model import ProtoIDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze ProtoIDS threshold operating point on validation set"
    )

    # Arguments must match those used in the original experiment
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="protoids_v1",
        help="Experiment name (matches model/results directories)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="ciciot2023",
        choices=["ciciot2023", "edgeiiot"],
        help="Dataset type",
    )

    parser.add_argument(
        "--withhold_open_set",
        action="store_true",
        help="True open-set: withhold MITM/VulnScan/BruteForce from TRAINING",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for data loading",
    )

    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Dimension of embedding space",
    )

    parser.add_argument(
        "--num_prototypes",
        type=int,
        default=3,
        help="Number of prototypes per class",
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate in encoder",
    )

    parser.add_argument(
        "--lambda_compact",
        type=float,
        default=0.1,
        help="Weight for compactness loss",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--ae_init",
        type=str,
        default="",
        help="Path to ae_pretrain.pth for encoder init",
    )

    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Use full 5.5M CICIOT23 train (not 235k dev parquet)",
    )

    parser.add_argument(
        "--threshold_steps",
        type=int,
        default=1000,
        help="Number of threshold steps to sweep",
    )

    parser.add_argument(
        "--threshold_min",
        type=float,
        default=None,
        help="Minimum threshold for sweep (auto-computed if None)",
    )

    parser.add_argument(
        "--threshold_max",
        type=float,
        default=None,
        help="Maximum threshold for sweep (auto-computed if None)",
    )

    return parser.parse_args()


def main():
    print("Starting threshold analysis...")
    args = parse_args()

    # ============================================================
    # SETUP
    # ============================================================

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ============================================================
    # PATHS
    # ============================================================

    data_dir = os.environ.get('EDGEIIOT_DATA_DIR', 'CICIOT23') if args.dataset == 'edgeiiot' else 'CICIOT23'
    artifact_dir = "results/baselines"

    development_subset_path = (
        None
        if args.full_data
        else "experiments/data/ciciot_dev.parquet"
    )

    model_save_dir = f"experiments/models/{args.experiment_name}"
    results_save_dir = f"experiments/results/{args.experiment_name}"

    # ============================================================
    # DETERMINE WITHHELD CLASSES
    # ============================================================

    withheld_for_train = None
    open_set = False

    if args.withhold_open_set:
        import json as _json

        if args.dataset == 'ciciot2023':
            label_mapping_path = os.path.join(
                artifact_dir,
                "ciciot_label_mapping.json",
            )
            with open(label_mapping_path) as f:
                label_mapping = _json.load(f)
            label_to_int = label_mapping["multiclass"]["label_to_int"]

            withheld_names = [
                "MITM-ArpSpoofing",
                "VulnerabilityScan",
                "DictionaryBruteForce",
            ]
            withheld_for_train = [
                label_to_int[name]
                for name in withheld_names
                if name in label_to_int
            ]
        else:
            # Edge-IIoTset
            label_mapping_path = os.path.join(
                "experiments", "results", args.experiment_name, "edgeiiot_label_mapping.json"
            )
            if not os.path.exists(label_mapping_path):
                label_mapping_path = os.path.join(
                    "experiments", "results", args.experiment_name, "ciciot_label_mapping.json"
                )

            with open(label_mapping_path) as f:
                label_mapping = _json.load(f)
            label_to_int = label_mapping["multiclass"]["label_to_int"]

            withheld_names = [
                "MITM",
                "Password",
                "Vulnerability_scanner",
            ]
            withheld_for_train = withheld_names  # pass names directly to dataset.py

        print(
            f"True open-set enabled: withholding {withheld_names} "
            f"-> indices {withheld_for_train} from TRAINING"
        )

        open_set = True

        # Avoid development subset leakage for open-set experiment
        development_subset_path = None

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("Loading data...")

    train_loader, val_loader, test_loader, num_classes, input_dim = (
        create_data_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            artifact_dir=artifact_dir,
            development_subset_path=development_subset_path,
            withheld_classes=withheld_for_train,
            open_set=open_set,
            scaler_save_dir=os.path.join(
                "experiments",
                "results",
                args.experiment_name,
            ),
            dataset_type=args.dataset,
        )
    )

    print(f"Validation batches: {len(val_loader)}")
    print(
        f"Test batches: {len(test_loader)} "
        "(not used for threshold selection)"
    )

    # ============================================================
    # GET UNKNOWN CLASS INDEX
    # ============================================================

    unknown_class_index = None

    if open_set:
        unknown_class_index = (
            train_loader.dataset.get_unknown_class_index()
        )

        print(
            f"Unknown class index: {unknown_class_index}"
        )

    # ============================================================
    # LOAD SAVED MODEL
    # ============================================================

    model_path = os.path.join(
        model_save_dir,
        "model.pth",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Have you run the experiment for "
            f"'{args.experiment_name}'?"
        )

    print(f"Loading model from {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = ProtoIDS(
        input_dim=checkpoint["protoids_args"]["input_dim"],
        num_classes=checkpoint["protoids_args"]["num_classes"],
        embedding_dim=checkpoint["protoids_args"]["embedding_dim"],
        num_prototypes_per_class=checkpoint[
            "protoids_args"
        ]["num_prototypes_per_class"],
        dropout_rate=checkpoint["protoids_args"]["dropout_rate"],
        unknown_class_index=checkpoint[
            "protoids_args"
        ].get("unknown_class_index", None),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # ============================================================
    # COLLECT VALIDATION DISTANCES
    # ============================================================

    all_min_dists = []
    all_true_labels = []
    all_pred_labels = []
    all_class_dists = []


    print("Collecting validation data...")

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass to get embeddings
            _, normalized_embedding, _, _ = model(batch_X)

            # Get raw class predictions and minimum
            # distances to known prototypes
            predicted_classes, min_distances = (
                model.predict_class(normalized_embedding)
            )
            #Minimum distance to each known class
            _, class_distances = model.prototype_layer(
                normalized_embedding
            )

            all_class_dists.append(
                class_distances.cpu()
            )

            all_min_dists.append(
                min_distances.cpu()
            )

            all_true_labels.append(
                batch_y.cpu()
            )

            all_pred_labels.append(
                predicted_classes.cpu()
            )

    # Concatenate all batches
    all_min_dists = torch.cat(
        all_min_dists
    ).numpy()

    all_true_labels = torch.cat(
        all_true_labels
    ).numpy()

    all_pred_labels = torch.cat(
        all_pred_labels
    ).numpy()

    all_class_dists = torch.cat(
        all_class_dists
    ).numpy()

    print(
        f"Collected per-class distances: "
        f"{all_class_dists.shape}"
    )
    print(
        f"Collected {len(all_min_dists)} "
        "validation samples"
    )

    # ============================================================
    # SEPARATE KNOWN AND UNKNOWN SAMPLES
    # ============================================================

    known_mask = (
        all_true_labels != unknown_class_index
    )

    unknown_mask = (
        all_true_labels == unknown_class_index
    )

    known_dists = all_min_dists[
        known_mask
    ]

    unknown_dists = all_min_dists[
        unknown_mask
    ]

    known_true_labels = all_true_labels[
        known_mask
    ]

    known_pred_labels = all_pred_labels[
        known_mask
    ]

    unknown_true_labels = all_true_labels[
        unknown_mask
    ]

    unknown_pred_labels = all_pred_labels[
        unknown_mask
    ]

    print(
        f"Known samples: {len(known_dists)}"
    )
    # ============================================================
    # PER-WITHHELD-CLASS DISTANCE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("PER-WITHHELD-CLASS DISTANCE ANALYSIS")
    print("=" * 60)

    if args.dataset == 'ciciot2023':
        val_csv_path = os.path.join(data_dir, "validation", "validation.csv")
        original_val_labels = []
        for chunk_df in pd.read_csv(val_csv_path, chunksize=100_000, usecols=["label"]):
            original_val_labels.extend(chunk_df["label"].tolist())
    else:
        # Edge-IIoTset: read the full CSV and extract validation split
        csv_path = os.environ.get('EDGEIIOT_DATA_DIR')
        if not csv_path or not os.path.isfile(csv_path):
            csv_path = os.path.join(data_dir, 'DNN-EdgeIIoT-dataset.csv')

        df = pd.read_csv(csv_path, usecols=["Attack_type"], low_memory=False)
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = df["Attack_type"].values
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_val_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=42)
        train_idx, val_idx = next(sss2.split(np.zeros(len(train_val_idx)), labels[train_val_idx]))
        val_idx = train_val_idx[val_idx]
        original_val_labels = labels[val_idx].tolist()

    original_val_labels = np.array(
        original_val_labels
    )

    if len(original_val_labels) != len(all_min_dists):
        raise RuntimeError(
            "Validation label count does not match "
            "distance count: "
            f"{len(original_val_labels)} vs "
            f"{len(all_min_dists)}"
        )


    print(
        f"Original validation labels loaded: "
        f"{len(original_val_labels)}"
    )

    print("\nWithheld-class distance distributions:")

    print(
        f"{'Class':<30}"
        f"{'Count':>10}"
        f"{'P50':>12}"
        f"{'P75':>12}"
        f"{'P90':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
        f"{'Mean':>12}"
    )

    for class_name in withheld_names:

        class_mask = (
            original_val_labels == class_name
        )

        class_dists = all_min_dists[class_mask]

        if len(class_dists) == 0:
            print(
                f"{class_name:<30}"
                f"{0:>10}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
                f"{'N/A':>12}"
            )
            continue

        p50, p75, p90, p95, p99 = np.percentile(
            class_dists,
            [50, 75, 90, 95, 99],
        )

        print(
            f"{class_name:<30}"
            f"{len(class_dists):>10}"
            f"{p50:>12.6f}"
            f"{p75:>12.6f}"
            f"{p90:>12.6f}"
            f"{p95:>12.6f}"
            f"{p99:>12.6f}"
            f"{np.mean(class_dists):>12.6f}"
        )

        print(
            f"  Range: "
            f"{np.min(class_dists):.6f} - "
            f"{np.max(class_dists):.6f}"
        )
    # ============================================================
    # PER-WITHHELD-CLASS KNOWN-CLASS DISTANCE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("PER-WITHHELD-CLASS KNOWN-CLASS DISTANCES")
    print("=" * 60)

    # all_class_dists has one column per known class. Each value is
    # the minimum cosine distance to that class's 3 prototypes.
    int_to_label = {
        int(index): name
        for name, index in label_to_int.items()
    }

    print(
        "\nClosest known classes by median prototype distance:"
    )

    for class_name in withheld_names:

        class_mask = (
            original_val_labels == class_name
        )

        class_matrix = all_class_dists[class_mask]

        if len(class_matrix) == 0:
            print(f"\n{class_name}: no samples found")
            continue

        # Median distance is more robust to outliers than the mean.
        class_medians = np.median(
            class_matrix,
            axis=0,
        )

        order = np.argsort(class_medians)

        print(f"\n{class_name}:")
        print(
            f"  {'Known class':<30}"
            f"{'Median':>12}"
            f"{'Mean':>12}"
            f"{'P90':>12}"
        )

        for class_idx in order[:10]:

            class_idx = int(class_idx)
            distances_to_class = class_matrix[:, class_idx]

            print(
                f"  {int_to_label.get(class_idx, f'Class_{class_idx}'):<30}"
                f"{np.median(distances_to_class):>12.6f}"
                f"{np.mean(distances_to_class):>12.6f}"
                f"{np.percentile(distances_to_class, 90):>12.6f}"
            )

    # ============================================================
    # UNKNOWN ATTACK ABSORPTION ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("UNKNOWN ATTACK ABSORPTION ANALYSIS")
    print("=" * 60)

    # Invert the original label mapping:
    # integer class index -> class name
    int_to_label = {
        int(index): name
        for name, index in label_to_int.items()
    }

    # Original validation labels are still available from the
    # per-withheld-class analysis above.
    print(
        "\nKnown-class predictions for each withheld attack:"
    )

    for class_name in withheld_names:

        class_mask = (
            original_val_labels == class_name
        )

        class_pred_labels = all_pred_labels[
            class_mask
        ]

        # Count predictions, excluding anything already represented
        # as UNKNOWN.
        known_predictions = (
            class_pred_labels[
                class_pred_labels != unknown_class_index
            ]
        )

        if len(known_predictions) == 0:
            print(
                f"\n{class_name}: "
                "No samples predicted as known."
            )
            continue

        unique_preds, counts = np.unique(
            known_predictions,
            return_counts=True,
        )

        # Sort from most frequent to least frequent.
        order = np.argsort(
            counts
        )[::-1]

        print(f"\n{class_name}:")
        print(
            f"  Total samples: "
            f"{len(class_pred_labels)}"
        )
        print(
            f"  Predicted as UNKNOWN before threshold: "
            f"{np.sum(class_pred_labels == unknown_class_index)}"
        )
        print(
            f"  Predicted as known class: "
            f"{len(known_predictions)}"
        )

        print("  Top known-class predictions:")

        for idx in order[:10]:

            predicted_index = int(
                unique_preds[idx]
            )

            predicted_name = int_to_label.get(
                predicted_index,
                f"Class_{predicted_index}",
            )

            count = int(
                counts[idx]
            )

            percentage = (
                100.0 * count / len(class_pred_labels)
            )

            print(
                f"    {predicted_name:<30} "
                f"{count:>8} "
                f"({percentage:>6.2f}%)"
            )

    print(
        f"Unknown samples: {len(unknown_dists)}"
    )

    # ============================================================
    # DISTANCE DISTRIBUTION ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print(
        "DISTANCE DISTRIBUTION ANALYSIS "
        "(VALIDATION SET)"
    )
    print("=" * 60)

    percentiles = [
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]

    print("\nMinimum prototype distance:")
    print(
        f"{'Percentile':<12}"
        f"{'Known':>12}"
        f"{'Unknown':>12}"
    )

    for p in percentiles:
        known_p = np.percentile(
            known_dists,
            p,
        )

        unknown_p = np.percentile(
            unknown_dists,
            p,
        )

        print(
            f"P{p:<10}"
            f"{known_p:>12.6f}"
            f"{unknown_p:>12.6f}"
        )

    print("\nSummary:")

    print(
        f"Known samples:   "
        f"{len(known_dists):,}"
    )

    print(
        f"Unknown samples: "
        f"{len(unknown_dists):,}"
    )

    print(
        f"\nKnown distance range:   "
        f"{known_dists.min():.6f} - "
        f"{known_dists.max():.6f}"
    )

    print(
        f"Unknown distance range: "
        f"{unknown_dists.min():.6f} - "
        f"{unknown_dists.max():.6f}"
    )

    print(
        f"\nKnown mean:   "
        f"{known_dists.mean():.6f}"
    )

    print(
        f"Unknown mean: "
        f"{unknown_dists.mean():.6f}"
    )

    print(
        f"Known median:   "
        f"{np.median(known_dists):.6f}"
    )

    print(
        f"Unknown median: "
        f"{np.median(unknown_dists):.6f}"
    )

    # ============================================================
    # DETERMINE THRESHOLD SWEEP RANGE
    # ============================================================

    if len(known_dists) == 0:
        raise ValueError(
            "No known samples found in validation set"
        )

    if args.threshold_min is None:
        th_min = np.min(known_dists)
    else:
        th_min = args.threshold_min

    if args.threshold_max is None:
        th_max = np.max(known_dists)
    else:
        th_max = args.threshold_max

    # Add small buffer to capture extremes
    th_range = th_max - th_min

    th_min = max(
        0.0,
        th_min - 0.05 * th_range,
    )

    th_max = (
        th_max + 0.05 * th_range
    )

    thresholds = np.linspace(
        th_min,
        th_max,
        args.threshold_steps,
    )

    print(
        f"Sweeping threshold from "
        f"{th_min:.4f} to {th_max:.4f} "
        f"in {args.threshold_steps} steps"
    )

    # ============================================================
    # ARRAYS TO STORE THRESHOLD METRICS
    # ============================================================

    # ARRAYS TO STORE THRESHOLD METRICS
    # ============================================================

    unknown_precisions = []
    unknown_recalls = []
    unknown_f1s = []
    unknown_fars = []
    unknown_krrs = []
    unknown_fdrs = []
    known_acceptance_rates = []
    known_macro_f1s = []

    # ============================================================
    # THRESHOLD SWEEP
    # ============================================================

    for T in thresholds:

        # Binary prediction:
        # 1 = unknown
        # 0 = known
        #
        # Prediction rule:
        # unknown if min_distance > T
        y_pred_binary = (
            all_min_dists > T
        ).astype(int)

        y_true_binary = (
            all_true_labels
            == unknown_class_index
        ).astype(int)

        # --------------------------------------------------------
        # Confusion matrix components
        # --------------------------------------------------------

        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))   # unknown predicted as unknown
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))   # unknown predicted as known
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))   # known predicted as unknown
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))   # known predicted as known

        # --------------------------------------------------------
        # Binary metrics
        # --------------------------------------------------------

        unknown_precision = (
            TP / (TP + FP)
            if (TP + FP) > 0
            else 0.0
        )

        unknown_recall = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else 0.0
        )

        unknown_f1 = (
            2 * unknown_precision * unknown_recall
            / (unknown_precision + unknown_recall)
            if (unknown_precision + unknown_recall) > 0
            else 0.0
        )

        unknown_far = (
            FN / (TP + FN)
            if (TP + FN) > 0
            else 0.0
        )

        unknown_krr = (
            FP / (FP + TN)
            if (FP + TN) > 0
            else 0.0
        )

        unknown_fdr = (
            FP / (TP + FP)
            if (TP + FP) > 0
            else 0.0
        )

        known_acceptance = (
            TN / (FP + TN)
            if (FP + TN) > 0
            else 0.0
        )

        # --------------------------------------------------------
        # Known Macro F1
        # Only true known samples accepted as known
        # --------------------------------------------------------

        true_known_accepted_mask = (
            (all_true_labels != unknown_class_index)
            & (all_min_dists <= T)
        )

        if np.sum(
            true_known_accepted_mask
        ) > 0:

            true_labels_known = (
                all_true_labels[
                    true_known_accepted_mask
                ]
            )

            pred_labels_known = (
                all_pred_labels[
                    true_known_accepted_mask
                ]
            )

            try:
                macro_f1 = f1_score(
                    true_labels_known,
                    pred_labels_known,
                    average="macro",
                    zero_division=0,
                )

            except Exception:
                macro_f1 = 0.0

        else:
            macro_f1 = 0.0

        # --------------------------------------------------------
        # Store metrics
        # --------------------------------------------------------

        unknown_precisions.append(
            unknown_precision
        )

        unknown_recalls.append(
            unknown_recall
        )

        unknown_f1s.append(
            unknown_f1
        )

        unknown_fars.append(
            unknown_far
        )

        unknown_krrs.append(
            unknown_krr
        )

        unknown_fdrs.append(
            unknown_fdr
        )

        known_acceptance_rates.append(
            known_acceptance
        )

        known_macro_f1s.append(
            macro_f1
        )
    # FIND BEST THRESHOLD
    # ============================================================

    unknown_f1s = np.array(unknown_f1s)

    best_idx = np.argmax(unknown_f1s)

    best_T = thresholds[
        best_idx
    ]

    best_f1 = unknown_f1s[
        best_idx
    ]

    best_precision = unknown_precisions[
        best_idx
    ]

    best_recall = unknown_recalls[
        best_idx
    ]

    best_far = unknown_fars[
        best_idx
    ]

    best_krr = unknown_krrs[
        best_idx
    ]

    best_known_acc = (
        known_acceptance_rates[
            best_idx
        ]
    )

    best_known_macro_f1 = (
        known_macro_f1s[
            best_idx
        ]
    )

    # ============================================================
    # OUTPUT BEST THRESHOLD
    # ============================================================

    print("\n" + "=" * 60)
    print(
        "THRESHOLD ANALYSIS RESULTS "
        "(VALIDATION SET)"
    )
    print("=" * 60)

    print(
        f"Threshold that maximizes "
        f"unknown F1: {best_T:.4f}"
    )

    print(
        f"  Unknown F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"  Unknown Precision: "
        f"{best_precision:.4f}"
    )

    print(
        f"  Unknown Recall: "
        f"{best_recall:.4f}"
    )

    print(
        f"  FAR: "
        f"{best_far:.4f}"
    )

    print(
        f"  KRR: "
        f"{best_krr:.4f}"
    )

    print(
        f"  Known Acceptance Rate: "
        f"{best_known_acc:.4f}"
    )

    print(
        f"  Known Macro F1: "
        f"{best_known_macro_f1:.4f}"
    )

    # ============================================================
    # TARGET FAR VALUES
    # ============================================================

    unknown_fars = np.array(unknown_fars)

    target_fars = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
    ]

    print("\n" + "-" * 60)
    print(
        "THRESHOLDS FOR TARGET FAR VALUES"
    )
    print("-" * 60)

    for target_far in target_fars:

        # Find threshold where FAR is closest
        # to target FAR
        far_errors = np.abs(
            unknown_fars - target_far
        )

        idx = np.argmin(
            far_errors
        )

        achieved_far = unknown_fars[
            idx
        ]

        T = thresholds[
            idx
        ]

        precision = unknown_precisions[
            idx
        ]

        recall = unknown_recalls[
            idx
        ]

        f1 = unknown_f1s[
            idx
        ]

        krr = unknown_krrs[
            idx
        ]

        known_acc = (
            known_acceptance_rates[
                idx
            ]
        )

        known_macro_f1 = (
            known_macro_f1s[
                idx
            ]
        )

        print(
            f"Target FAR "
            f"{target_far:.0%}:"
        )

        print(
            f"  Threshold: "
            f"{T:.4f}"
        )

        print(
            f"  Achieved FAR: "
            f"{achieved_far:.4f}"
        )

        print(
            f"  Unknown Precision: "
            f"{precision:.4f}"
        )

        print(
            f"  Unknown Recall: "
            f"{recall:.4f}"
        )

        print(
            f"  Unknown F1: "
            f"{f1:.4f}"
        )

        print(
            f"  KRR: "
            f"{krr:.4f}"
        )

        print(
            f"  Known Acceptance Rate: "
            f"{known_acc:.4f}"
        )

        print(
            f"  Known Macro F1: "
            f"{known_macro_f1:.4f}"
        )

        print()

    # ============================================================
    # CURRENT EXPERIMENT THRESHOLD
    # ============================================================

    threshold_path = os.path.join(
        model_save_dir,
        "threshold.npy",
    )

    if os.path.exists(
        threshold_path
    ):

        current_threshold = np.load(
            threshold_path
        )[0]

        print("-" * 60)

        print(
            f"Current threshold from "
            f"experiment: "
            f"{current_threshold:.4f}"
        )

        # Find closest threshold
        # in our sweep
        th_errors = np.abs(
            thresholds
            - current_threshold
        )

        idx = np.argmin(
            th_errors
        )

        print(
            "At this threshold:"
        )

        print(
            f"  Unknown F1: "
            f"{unknown_f1s[idx]:.4f}"
        )

        print(
            f"  FAR: "
            f"{unknown_fars[idx]:.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
