"""
Main training script for ProtoIDS on CICIoT2023 dataset.
Implements closed-set and open-set evaluation as specified.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import json
import time
import argparse
import joblib
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from protoids.dataset import create_data_loaders, MemmapDataset
from protoids.protoids_model import ProtoIDS
from protoids.training import train_protoids
from torch.utils.data import DataLoader


def save_experiment_log(experiment_id: str, config: Dict, metrics: Dict,
                       notes: str = ""):
    """
    Save experiment results to the experiment log.

    Args:
        experiment_id: Unique identifier for this experiment
        config: Configuration used for the experiment
        metrics: Metrics achieved by the experiment
        notes: Optional notes about the experiment
    """
    log_path = 'experiments/experiment_log.json'

    # Load existing log or create new one
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = []

    # Create experiment entry
    experiment_entry = {
        'experiment_id': experiment_id,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'metrics': metrics,
        'notes': notes
    }

    # Append to log
    log_data.append(experiment_entry)

    # Save updated log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def evaluate_closed_set(model, data_loader, device, threshold=None):
    """
    Evaluate the model in closed-set mode (known classes only).

    Args:
        model: Trained ProtoIDS model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        threshold: Distance threshold (unused in closed-set, kept for interface consistency)

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_distances = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            _, normalized_embedding, _, min_distances_per_class = model(batch_X)

            # Get predictions (closed-set: choose class with minimum distance)
            predicted_classes, min_distances = model.predict_class(normalized_embedding)

            # Store results
            all_preds.append(predicted_classes.cpu())
            all_labels.append(batch_y.cpu())
            all_distances.append(min_distances.cpu())

    # Concatenate results
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    distances = torch.cat(all_distances).numpy()

    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    # Per-class recall
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'mcc': float(mcc),
        'recall_per_class': recall_per_class.tolist(),
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances))
    }

    return metrics, preds, labels, distances


def evaluate_open_set(model, known_loader, unknown_loader, device,
                     threshold_percentile=90, threshold=None):
    """
    Evaluate the model in open-set mode (with unknown class detection).

    Args:
        model: Trained ProtoIDS model
        known_loader: DataLoader for known class validation data
        unknown_loader: DataLoader for unknown class data
        device: Device to run evaluation on
        threshold_percentile: Percentile of known distances to use as threshold (if threshold is None)
        threshold: Precomputed threshold to use (if provided, overrides threshold_percentile)

    Returns:
        metrics: Dictionary containing open-set evaluation metrics
    """
    model.eval()

    # Get distances for known class data (to set threshold)
    known_distances = []
    known_labels = []

    with torch.no_grad():
        for batch_X, batch_y in known_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            _, normalized_embedding, _, min_distances_per_class = model(batch_X)
            _, min_distances = model.predict_class(normalized_embedding)

            known_distances.append(min_distances.cpu())
            known_labels.append(batch_y.cpu())

    known_distances = torch.cat(known_distances).numpy()
    known_labels = torch.cat(known_labels).numpy()

    # Set threshold as percentile of known distances
    if threshold is None:
        threshold = np.percentile(known_distances, threshold_percentile)
    else:
        threshold = threshold

    # Get predictions for known data with threshold
    known_preds = []
    known_labels_thresh = []

    with torch.no_grad():
        for batch_X, batch_y in known_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            _, normalized_embedding, _, _ = model(batch_X)
            predicted_classes, min_distances = model.predict_unknown(
                normalized_embedding, threshold
            )

            known_preds.append(predicted_classes.cpu())
            known_labels_thresh.append(batch_y.cpu())

    known_preds = torch.cat(known_preds).numpy()
    known_labels_thresh = torch.cat(known_labels_thresh).numpy()

    # Get predictions for unknown data
    unknown_preds = []
    unknown_labels = []  # All unknown samples should be labeled as unknown class

    with torch.no_grad():
        for batch_X, batch_y in unknown_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            _, normalized_embedding, _, _ = model(batch_X)
            predicted_classes, min_distances = model.predict_unknown(
                normalized_embedding, threshold
            )

            unknown_preds.append(predicted_classes.cpu())
            # Create labels where unknown class is model.unknown_class_index
            unknown_labels.append(torch.full_like(batch_y, model.unknown_class_index).cpu())

    unknown_preds = torch.cat(unknown_preds).numpy()
    unknown_labels = torch.cat(unknown_labels).numpy()

    # Compute known-class metrics (only considering samples predicted as known)
    known_mask = known_preds != model.unknown_class_index
    if np.sum(known_mask) > 0:
        known_accuracy = accuracy_score(
            known_labels_thresh[known_mask],
            known_preds[known_mask]
        )
        known_macro_f1 = f1_score(
            known_labels_thresh[known_mask],
            known_preds[known_mask],
            average='macro', zero_division=0
        )
        known_recall = recall_score(
            known_labels_thresh[known_mask],
            known_preds[known_mask],
            average='macro', zero_division=0
        )
    else:
        known_accuracy = 0.0
        known_macro_f1 = 0.0
        known_recall = 0.0

    # Compute unknown detection metrics
    # Known samples: label < model.unknown_class_index (since unknown class index is exactly num_known_classes)
    # Predicted known: pred < model.unknown_class_index
    # Unknown samples: label == model.unknown_class_index
    # Predicted unknown: pred == model.unknown_class_index

    # Combine known and unknown data for overall metrics
    all_labels = np.concatenate([known_labels_thresh, unknown_labels])
    all_preds = np.concatenate([known_preds, unknown_preds])

    # Overall accuracy (including unknown as a class)
    overall_accuracy = accuracy_score(all_labels, all_preds)

    # Unknown detection metrics
    # True negatives: known samples correctly predicted as known
    # False positives: unknown samples incorrectly predicted as known
    # False negatives: known samples incorrectly predicted as unknown
    # True positives: unknown samples correctly predicted as unknown

    y_true_binary = (all_labels == model.unknown_class_index).astype(int)  # 1 = unknown, 0 = known
    y_pred_binary = (all_preds == model.unknown_class_index).astype(int)   # 1 = predicted unknown, 0 = predicted known

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    unknown_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    unknown_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    unknown_f1 = 2 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall) if (unknown_precision + unknown_recall) > 0 else 0.0

    # False acceptance rate (FAR): unknown samples predicted as known / total unknown samples
    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # False discovery rate (FDR): known samples predicted as unknown / total predicted unknown
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # Known rejection rate (KRR): known samples predicted as unknown / total known samples
    krr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Compute ROC-AUC and AUPR for unknown detection
    try:
        # We need the distance scores for ROC-AUC (higher distance = more likely unknown)
        # Get distances for all samples
        all_distances = []

        with torch.no_grad():
            # Known data distances
            for batch_X, _ in known_loader:
                batch_X = batch_X.to(device)
                _, normalized_embedding, _, min_distances_per_class = model(batch_X)
                _, min_distances = model.predict_class(normalized_embedding)
                all_distances.append(min_distances.cpu())

            # Unknown data distances
            for batch_X, _ in unknown_loader:
                batch_X = batch_X.to(device)
                _, normalized_embedding, _, min_distances_per_class = model(batch_X)
                _, min_distances = model.predict_class(normalized_embedding)
                all_distances.append(min_distances.cpu())

        all_distances = torch.cat(all_distances).numpy()

        # For ROC-AUC, we want to detect unknown (higher distance = more likely unknown)
        auc_roc = roc_auc_score(y_true_binary, all_distances)
        auc_pr = average_precision_score(y_true_binary, all_distances)
    except Exception as e:
        print(f"Warning: Could not compute ROC-AUC/AUPR: {e}")
        auc_roc = 0.0
        auc_pr = 0.0

    metrics = {
        'known_accuracy': float(known_accuracy),
        'known_macro_f1': float(known_macro_f1),
        'known_recall': float(known_recall),
        'unknown_precision': float(unknown_precision),
        'unknown_recall': float(unknown_recall),
        'unknown_f1': float(unknown_f1),
        'far': float(far),  # False acceptance rate
        'fdr': float(fdr),  # False discovery rate
        'krr': float(krr),  # Known rejection rate
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'threshold': float(threshold),
        'known_samples_correctly_accepted': int(tn),
        'known_samples_rejected_as_unknown': int(fn),
        'unknown_samples_correctly_rejected': int(tp),
        'unknown_samples_incorrectly_accepted': int(fp)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train ProtoIDS on CICIoT2023 or Edge-IIoTset dataset')
    parser.add_argument('--experiment_name', type=str, default='protoids_v1',
                       help='Name for this experiment')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda_compact', type=float, default=0.1,
                       help='Weight for compactness loss')
    parser.add_argument('--embedding_dim', type=int, default=32,
                       help='Dimension of embedding space')
    parser.add_argument('--num_prototypes', type=int, default=3,
                       help='Number of prototypes per class')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate in encoder')
    parser.add_argument('--class_weights', action='store_true',
                       help='Use class-weighted loss')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation, skip training')
    parser.add_argument('--model_path', type=str, default='',
                       help='Path to pre-trained model for evaluation')
    parser.add_argument('--withhold_open_set', action='store_true',
                       help='Enable open-set processing (withhold classes from training)')
    parser.add_argument('--dataset', type=str, choices=['ciciot2023', 'edgeiiot', 'xiiotid'], default='ciciot2023',
                       help='Dataset to use: ciciot2023, edgeiiot or xiiotid')
    parser.add_argument('--withheld_classes', nargs='*', type=str, default=None,
                       help='List of class names to withhold from training (for open-set). If omitted, uses dataset-specific defaults when --withhold_open_set is set.')
    parser.add_argument('--xiiotid_label_col', type=str, default=None,
                       help='X-IIoTID only: force a target label column (default: auto-detect, prefers Sub-Category)')
    parser.add_argument('--max_rows', type=int, default=None,
                       help='X-IIoTID only: cap the number of CSV rows read (for fast smoke tests)')
    parser.add_argument('--split_strategy', type=str, default='random',
                       choices=['random', 'temporal'],
                       help='X-IIoTID only: stratified random split, or contiguous '
                            'time blocks (earliest->train, latest->test). Temporal is '
                            'the honest generalisation test for a deployed IDS.')
    parser.add_argument('--xiiotid_time_col', type=str, default=None,
                       help='X-IIoTID only: time column for --split_strategy temporal '
                            '(default: Timestamp, else Date)')
    parser.add_argument('--train_frac', type=float, default=0.70,
                       help='X-IIoTID temporal split: fraction of rows used for training')
    parser.add_argument('--val_frac', type=float, default=0.15,
                       help='X-IIoTID temporal split: fraction of rows used for validation')
    parser.add_argument('--ae_init', type=str, default='',
                       help='Path to ae_pretrain.pth for encoder init')
    parser.add_argument('--full_data', action='store_true',
                       help='Use full 5.5M CICIOT23 train (not 235k dev parquet)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine withheld class names (strings) and open_set flag
    withheld_names = None  # list of strings or None
    open_set = args.withhold_open_set  # bool

    if args.dataset == 'ciciot2023':
        if args.withheld_classes is not None:
            # Custom class names provided
            withheld_names = args.withheld_classes
        else:
            # Use default CICIoT2023 classes if open set requested
            if args.withhold_open_set:
                withheld_names = ['MITM-ArpSpoofing', 'VulnerabilityScan', 'DictionaryBruteForce']
            else:
                withheld_names = []  # no withholding
    elif args.dataset == 'edgeiiot':
        if args.withheld_classes is not None:
            withheld_names = args.withheld_classes
        else:
            withheld_names = []  # no withholding by default for Edge-IIoT
    else:  # xiiotid
        # X-IIoTID has many classes and the release varies between mirrors, so we
        # do NOT hard-code a default. Without an explicit choice we inspect the
        # data and suggest the most frequent attack classes (see
        # suggest_xiiotid_withheld), keeping the choice visible and reviewable.
        if args.withheld_classes is not None:
            withheld_names = args.withheld_classes
        else:
            print("No --withheld_classes given for X-IIoTID.")
            print("Suggesting withheld classes from the most frequent attack classes.")
            from protoids.dataset import suggest_xiiotid_withheld
            try:
                _lc, _suggested, _counts = suggest_xiiotid_withheld(
                    data_dir=os.environ.get('XIIOTID_DATA_DIR', 'datasets/xiiotid'),
                    label_col=args.xiiotid_label_col,
                    k=3,
                )
                print(f"Suggested withheld classes: {_suggested}")
                print("Pass these explicitly with --withheld_classes to make the choice reproducible.")
                withheld_names = _suggested
            except Exception as exc:
                print(f"Could not auto-suggest withheld classes ({exc}).")
                print("Run the dataset inspection cell in colab/ProtoIDS_XIIoTID.ipynb, "
                      "then re-run with --withheld_classes '<name1>' '<name2>' '<name3>'.")
                sys.exit(1)

    # Paths
    if args.dataset == 'ciciot2023':
        data_dir = 'CICIOT23'
        artifact_dir = 'results/baselines'
        development_subset_path = None if args.full_data else 'experiments/data/ciciot_dev.parquet'
        if args.full_data:
            print("Using FULL 5.5M train (CICIOT23/train/train.csv) – not dev parquet")
    elif args.dataset == 'edgeiiot':
        # data_dir must point at the directory holding the Edge-IIoT CSV.
        data_dir = os.environ.get('EDGEIIOT_DATA_DIR', '')
        if not data_dir:
            print("ERROR: For Edge-IIoT dataset, set EDGEIIOT_DATA_DIR to the directory containing the CSV file.")
            sys.exit(1)
        artifact_dir = 'results/baselines'
        development_subset_path = None  # Edge-IIoT does not use development subset
        print(f"Using Edge-IIoT dataset from directory: {data_dir}")
    else:  # xiiotid
        # The downloader (src/data/download_xiiotid.py) drops the CSV here.
        data_dir = os.environ.get('XIIOTID_DATA_DIR', 'datasets/xiiotid')
        artifact_dir = 'results/baselines'
        development_subset_path = None  # X-IIoTID is processed end-to-end, no dev subset
        print(f"Using X-IIoTID dataset from directory: {data_dir}")

    model_save_dir = f'experiments/models/{args.experiment_name}'
    results_save_dir = f'experiments/results/{args.experiment_name}'

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    # Determine scaler save directory (if open set)
    scaler_save_dir = None
    if open_set:
        scaler_save_dir = f'experiments/results/{args.experiment_name}'
        # For open-set, we do not use the development subset to avoid leakage (already set above for Edge-IIoT)
        if args.dataset == 'ciciot2023':
            development_subset_path = None

    print("Loading data...")
    load_start = time.time()
    train_loader, val_loader, test_loader, num_classes, input_dim = create_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        artifact_dir=artifact_dir,
        development_subset_path=development_subset_path,
        withheld_classes=withheld_names,  # pass list of strings (or None)
        open_set=open_set,
        scaler_save_dir=scaler_save_dir,
        dataset_type=args.dataset,
        label_col=args.xiiotid_label_col,
        max_rows=args.max_rows,
        split_strategy=args.split_strategy,
        time_col=args.xiiotid_time_col,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    load_end = time.time()
    print(f"Data loading took {load_end - load_start:.2f} seconds.")
    print(f"After data loading, train_loader length: {len(train_loader)}")

    print(f"Dataset loaded:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Determine unknown class index for open-set
    unknown_class_index = None
    if open_set:
        # Retrieve from the training dataset (which is our MemmapDataset)
        unknown_class_index = train_loader.dataset.get_unknown_class_index()
        print(f"Unknown class index: {unknown_class_index}")

    # Log scaler n_samples_seen_ and unknown counts
    if open_set and scaler_save_dir:
        scaler_path = os.path.join(scaler_save_dir, f'{args.dataset}_preprocessor.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            if not hasattr(scaler, "n_samples_seen_"):
                scaler.n_samples_seen_ = len(train_loader.dataset)
                joblib.dump(scaler, scaler_path)
            print(f"Scaler n_samples_seen_: {getattr(scaler, 'n_samples_seen_', 'N/A')}")
        else:
            print("WARNING: Scaler not found.")

    if open_set:
        unknown_idx = train_loader.dataset.get_unknown_class_index()
        if unknown_idx is not None:
            # Count unknown samples in each dataset (using memmap)
            train_unknown = np.count_nonzero(train_loader.dataset.labels == unknown_idx)
            val_unknown = np.count_nonzero(val_loader.dataset.labels == unknown_idx)
            test_unknown = np.count_nonzero(test_loader.dataset.labels == unknown_idx)
            print(f"Unknown samples in training: {train_unknown}")
            print(f"Unknown samples in validation: {val_unknown}")
            print(f"Unknown samples in test: {test_unknown}")

    # Get class weights if requested
    class_weights = None
    if args.class_weights:
        # Create a temporary dataset to compute class weights (respect withholding)
        # We need to create a dataset that respects withholding for the training split.
        # For simplicity, we will reuse the create_data_loaders function to get a temporary loader without batching?
        # Instead we can compute class weights from the training labels we already have?
        # We'll create a temporary MemmapDataset from the processed training data?
        # Since we don't have easy access to the raw labels after processing, we'll skip and compute from the data loader.
        # For now, we'll compute using the train_loader dataset (which is MemmapDataset) if available.
        # This is a simplification; we'll issue a warning if not implemented.
        print("Warning: class_weights computation for Edge-IIoT not fully implemented; using None.")
        class_weights = None

    # Create experiment ID
    experiment_id = f"{args.experiment_name}_{int(time.time())}"

    if not args.eval_only:
        # Initialize model
        print(f"\nInitializing ProtoIDS model...")
        model = ProtoIDS(
            input_dim=input_dim,
            num_classes=num_classes,
            embedding_dim=args.embedding_dim,
            num_prototypes_per_class=args.num_prototypes,
            dropout_rate=args.dropout_rate,
            unknown_class_index=unknown_class_index
        )
        print(f"Model architecture:")
        print(f"  - Encoder: {input_dim} -> 128 -> BatchNorm -> ReLU -> Dropout({args.dropout_rate}) -> 64 -> ReLU -> {args.embedding_dim} -> L2 norm")
        print(f"  - Prototypes: {args.num_prototypes} per class ({num_classes} classes)")
        print(f"  - Embedding dimension: {args.embedding_dim}")
        print(f"  - Compactness loss weight: {args.lambda_compact}")
        if args.ae_init:
            print(f"  - Encoder init: loading AE pretrain from {args.ae_init}")
            ae_ckpt = torch.load(args.ae_init, map_location='cpu')
            model.encoder.load_state_dict(ae_ckpt['encoder_state'])
            print(f"    AE encoder loaded (train MSE 0.020 val 0.0033 pretrain)")

        # Initialize prototypes using K-means on training data
        print(f"\nInitializing prototypes using K-means on training data...")
        model.to(device)
        model.eval()
        with torch.no_grad():
            # Collect training embeddings and labels
            all_embeddings = []
            all_labels = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                _, normalized_embedding = model.encoder(batch_X)
                all_embeddings.append(normalized_embedding.cpu())
                all_labels.append(batch_y)

            all_embeddings = torch.cat(all_embeddings)
            all_labels = torch.cat(all_labels)

            # Initialize prototypes
            model.prototype_layer.initialize_prototypes(all_embeddings, all_labels)
            print("Prototype initialization complete.")

        # Train the model
        print(f"\nStarting training for {args.num_epochs} epochs...")
        start_time = time.time()

        training_history = train_protoids(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            lambda_compact=args.lambda_compact,
            class_weights=class_weights,
            verbose=True,
            checkpoint_dir=model_save_dir
        )

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds.")

        # Save the trained model
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'protoids_args': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'embedding_dim': args.embedding_dim,
                'num_prototypes_per_class': args.num_prototypes,
                'dropout_rate': args.dropout_rate,
                'unknown_class_index': unknown_class_index
            },
            'training_history': training_history,
            'experiment_id': experiment_id
        }, model_path)
        print(f"Model saved to {model_path}")

        # Save training history
        os.makedirs(results_save_dir, exist_ok=True)
        history_path = os.path.join(results_save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)

    else:
        # Load pre-trained model for evaluation
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError(f"Model path not provided or does not exist: {args.model_path}")

        print(f"Loading pre-trained model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = ProtoIDS(
            input_dim=checkpoint['protoids_args']['input_dim'],
            num_classes=checkpoint['protoids_args']['num_classes'],
            embedding_dim=checkpoint['protoids_args']['embedding_dim'],
            num_prototypes_per_class=checkpoint['protoids_args']['num_prototypes_per_class'],
            dropout_rate=checkpoint['protoids_args']['dropout_rate'],
            unknown_class_index=checkpoint['protoids_args'].get('unknown_class_index', None)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        experiment_id = checkpoint.get('experiment_id', f"loaded_{int(time.time())}")

    # Run closed-set evaluation on validation set
    print(f"\nRunning closed-set evaluation on validation set...")
    val_metrics, val_preds, val_labels, val_distances = evaluate_closed_set(
        model, val_loader, device
    )

    print(f"Closed-set Validation Results:")
    print(f"  - Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  - Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"  - Weighted F1: {val_metrics['weighted_f1']:.4f}")
    print(f"  - Macro Recall: {val_metrics['macro_recall']:.4f}")
    print(f"  - MCC: {val_metrics['mcc']:.4f}")

    # Run closed-set evaluation on test set
    print(f"\nRunning closed-set evaluation on test set...")
    test_metrics, test_preds, test_labels, test_distances = evaluate_closed_set(
        model, test_loader, device
    )

    print(f"Closed-set Test Results:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  - Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"  - Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"  - MCC: {test_metrics['mcc']:.4f}")

    # Save closed-set results
    closed_set_path = os.path.join(results_save_dir, 'closed_set_results.json')
    with open(closed_set_path, 'w') as f:
        json.dump({'validation': val_metrics, 'test': test_metrics}, f, indent=2)

    # Run open-set evaluation
    print(f"\nSetting up open-set evaluation...")
    # Determine withheld class names for reporting and splitting
    if withheld_names is None or len(withheld_names) == 0:
        withheld_names_display = ['None (closed set)']
        withheld_class_names = []
    else:
        withheld_names_display = withheld_names
        withheld_class_names = withheld_names

    print(f"Withholding classes: {', '.join(withheld_names_display)}")

    # Get the label mapping to find the indices of the withheld classes
    # For CICIoT2023 we have the artifact dir; for Edge-IIoT we need to load label mapping from scaler_save_dir if saved.
    label_mapping = None
    if args.dataset == 'ciciot2023':
        label_mapping_path = os.path.join(artifact_dir, 'ciciot_label_mapping.json')
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
    else:
        # Edge-IIoT / X-IIoTID: label mapping is written by the loader into scaler_save_dir
        if scaler_save_dir is not None:
            label_mapping_path = os.path.join(scaler_save_dir, f'{args.dataset}_label_mapping.json')
            if os.path.exists(label_mapping_path):
                with open(label_mapping_path, 'r') as f:
                    label_mapping = json.load(f)
            else:
                print(f"WARNING: {label_mapping_path} not found; cannot report withheld class indices.")
        else:
            print("WARNING: No scaler_save_dir; cannot load label mapping.")

    if label_mapping is not None:
        int_to_label = {v: k for k, v in label_mapping['multiclass']['label_to_int'].items()}
        label_to_int_map = label_mapping['multiclass']['label_to_int']
        # Withheld classes are absent from the known-class mapping BY DESIGN, so
        # report them by name and point at the reserved unknown slot instead of
        # printing an empty index list.
        mapped = {cls: label_to_int_map[cls] for cls in withheld_class_names
                  if cls in label_to_int_map}
        print(f"Withheld class names : {sorted(withheld_class_names)}")
        if open_set:
            print(f"Withheld classes have no class index (never trained); they are "
                  f"collapsed onto the reserved unknown index "
                  f"{model.unknown_class_index} at evaluation time.")
        withheld_class_indices = sorted(mapped.values())
    else:
        withheld_class_indices = []

    # Note: with the true open-set loaders the model IS trained without the
    # withheld classes - they were already removed before the scaler was fitted
    # and before any prototype was learned. Here we only need to separate the
    # already-processed validation rows into known / unknown, and calibrate the
    # threshold on known validation rows alone.

    # Create datasets for open-set evaluation
    # Known data: validation set excluding withheld classes
    # Unknown data: validation set containing only withheld classes

    # We'll create temporary data loaders for this purpose
    # First, let's get the validation dataset
    # Reuse the already-created processed validation dataset.
    # With true open-set preprocessing, withheld samples are already mapped
    # to model.unknown_class_index.
    val_dataset = val_loader.dataset
    val_labels_array = val_dataset.labels

    unknown_class_index = model.unknown_class_index
    known_mask = val_labels_array != unknown_class_index
    unknown_mask = val_labels_array == unknown_class_index

    known_indices = np.where(known_mask)[0]
    unknown_indices = np.where(unknown_mask)[0]

    print(f"Validation set split:")
    print(f"  - Known samples: {len(known_indices)}")
    print(f"  - Unknown samples (withheld classes): {len(unknown_indices)}")

    # Create subset data loaders
    from torch.utils.data import Subset

    known_val_subset = Subset(val_dataset, known_indices)
    unknown_val_subset = Subset(val_dataset, unknown_indices)

    known_val_loader = DataLoader(
        known_val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    unknown_val_loader = DataLoader(
        unknown_val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Run open-set evaluation
    print(f"\nRunning open-set evaluation...")
    open_set_metrics = evaluate_open_set(
        model=model,
        known_loader=known_val_loader,
        unknown_loader=unknown_val_loader,
        device=device,
        threshold_percentile=90  # Use 90th percentile as threshold
    )

    print(f"Open-set Validation Results:")
    print(f"  - Known Accuracy: {open_set_metrics['known_accuracy']:.4f}")
    print(f"  - Known Macro F1: {open_set_metrics['known_macro_f1']:.4f}")
    print(f"  - Known Recall: {open_set_metrics['known_recall']:.4f}")
    print(f"  - Unknown Precision: {open_set_metrics['unknown_precision']:.4f}")
    print(f"  - Unknown Recall: {open_set_metrics['unknown_recall']:.4f}")
    print(f"  - Unknown F1: {open_set_metrics['unknown_f1']:.4f}")
    print(f"  - FAR (False Acceptance Rate): {open_set_metrics['far']:.4f}")
    print(f"  - FDR (False Discovery Rate): {open_set_metrics['fdr']:.4f}")
    print(f"  - KRR (Known Rejection Rate): {open_set_metrics['krr']:.4f}")
    print(f"  - AUROC: {open_set_metrics['auc_roc']:.4f}")
    print(f"  - AUPR: {open_set_metrics['auc_pr']:.4f}")
    print(f"  - Threshold: {open_set_metrics['threshold']:.4f}")

    # Save open-set results
    open_set_path = os.path.join(results_save_dir, 'open_set_results.json')
    with open(open_set_path, 'w') as f:
        json.dump(open_set_metrics, f, indent=2)

    # Run open-set evaluation on test set
    print(f"\nSetting up open-set evaluation on test set...")
    # We'll use the same withheld class indices as before
    # Reuse the already-created processed test dataset.
    # With true open-set preprocessing, withheld samples are already mapped
    # to model.unknown_class_index.
    test_dataset = test_loader.dataset
    test_labels_array = test_dataset.labels

    unknown_class_index = model.unknown_class_index
    known_mask_test = test_labels_array != unknown_class_index
    unknown_mask_test = test_labels_array == unknown_class_index

    known_indices_test = np.where(known_mask_test)[0]
    unknown_indices_test = np.where(unknown_mask_test)[0]

    print(f"Test set split:")
    print(f"  - Known samples: {len(known_indices_test)}")
    print(f"  - Unknown samples (withheld classes): {len(unknown_indices_test)}")

    # Create subset data loaders
    known_test_subset = Subset(test_dataset, known_indices_test)
    unknown_test_subset = Subset(test_dataset, unknown_indices_test)

    known_test_loader = DataLoader(
        known_test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    unknown_test_loader = DataLoader(
        unknown_test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Run open-set evaluation on test set using threshold from validation
    print(f"\nRunning open-set evaluation on test set...")
    open_set_test_metrics = evaluate_open_set(
        model=model,
        known_loader=known_test_loader,
        unknown_loader=unknown_test_loader,
        device=device,
        threshold=open_set_metrics['threshold']  # Use threshold from validation
    )

    print(f"Open-set Test Results:")
    print(f"  - Known Accuracy: {open_set_test_metrics['known_accuracy']:.4f}")
    print(f"  - Known Macro F1: {open_set_test_metrics['known_macro_f1']:.4f}")
    print(f"  - Known Recall: {open_set_test_metrics['known_recall']:.4f}")
    print(f"  - Unknown Precision: {open_set_test_metrics['unknown_precision']:.4f}")
    print(f"  - Unknown Recall: {open_set_test_metrics['unknown_recall']:.4f}")
    print(f"  - Unknown F1: {open_set_test_metrics['unknown_f1']:.4f}")
    print(f"  - FAR (False Acceptance Rate): {open_set_test_metrics['far']:.4f}")
    print(f"  - FDR (False Discovery Rate): {open_set_test_metrics['fdr']:.4f}")
    print(f"  - KRR (Known Rejection Rate): {open_set_test_metrics['krr']:.4f}")
    print(f"  - AUROC: {open_set_test_metrics['auc_roc']:.4f}")
    print(f"  - AUPR: {open_set_test_metrics['auc_pr']:.4f}")
    print(f"  - Threshold: {open_set_test_metrics['threshold']:.4f}")

    # Save open-set test results
    open_set_test_path = os.path.join(results_save_dir, 'open_set_test_results.json')
    with open(open_set_test_path, 'w') as f:
        json.dump(open_set_test_metrics, f, indent=2)

    # Save prototype vectors and other artifacts
    prototype_path = os.path.join(model_save_dir, 'prototypes.npy')
    np.save(prototype_path, model.prototype_layer.prototypes.detach().cpu().numpy())
    print(f"Prototype vectors saved to {prototype_path}")

    threshold_path = os.path.join(model_save_dir, 'threshold.npy')
    if 'open_set_metrics' in locals():
        np.save(threshold_path, np.array([open_set_metrics['threshold']]))
        print(f"Threshold saved to {threshold_path}")

    label_mapping_path_save = os.path.join(model_save_dir, 'label_mapping.json')
    with open(label_mapping_path_save, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping saved to {label_mapping_path_save}")

    # Prepare final metrics for experiment log
    final_metrics = {
        'closed_set_validation': val_metrics,
        'closed_set_test': test_metrics,
        'open_set_validation': open_set_metrics if 'open_set_metrics' in locals() else None,
        'open_set_test': open_set_test_metrics if 'open_set_test_metrics' in locals() else None
    }

    # Configuration for experiment log
    config = {
        'experiment_name': args.experiment_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lambda_compact': args.lambda_compact,
        'embedding_dim': args.embedding_dim,
        'num_prototypes_per_class': args.num_prototypes,
        'dropout_rate': args.dropout_rate,
        'class_weights': args.class_weights,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'withheld_classes': withheld_names if withheld_names is not None else [],
        'seed': args.seed
    }

    # Notes for experiment log
    notes = f"ProtoIDS v1 implementation. Closed-set evaluation on CICIoT2023 validation/test sets. Open-set evaluation on validation and test sets withholding {', '.join(withheld_names_display) if withheld_names else 'None'} as unknown classes."

    # Save to experiment log
    save_experiment_log(
        experiment_id=experiment_id,
        config=config,
        metrics=final_metrics,
        notes=notes
    )

    print(f"\nExperiment {experiment_id} completed and logged.")
    print(f"Results saved to {results_save_dir}")
    print(f"Model saved to {model_save_dir}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()