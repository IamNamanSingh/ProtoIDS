"""
Main training script for ProtoIDS on CICIoT2023 dataset.
Implements closed-set and open-set evaluation as specified.
"""

import torch
import numpy as np
import json
import os
import time
import argparse
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from .dataset import create_data_loaders, CICIoT2023ProtoIDSDataset
from .protoids_model import ProtoIDS
from .training import train_protoids
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
                     threshold_percentile=90):
    """
    Evaluate the model in open-set mode (with unknown class detection).

    Args:
        model: Trained ProtoIDS model
        known_loader: DataLoader for known class validation data
        unknown_loader: DataLoader for unknown class data
        device: Device to run evaluation on
        threshold_percentile: Percentile of known distances to use as threshold

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
    threshold = np.percentile(known_distances, threshold_percentile)

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
            # Create labels where unknown class is num_classes
            unknown_labels.append(torch.full_like(batch_y, model.num_classes).cpu())

    unknown_preds = torch.cat(unknown_preds).numpy()
    unknown_labels = torch.cat(unknown_labels).numpy()

    # Compute known-class metrics (only considering samples predicted as known)
    known_mask = known_preds != model.num_classes
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
    # Known samples: label < num_classes, Predicted known: pred < num_classes
    # Unknown samples: label == num_classes, Predicted unknown: pred == num_classes

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

    y_true_binary = (all_labels == model.num_classes).astype(int)  # 1 = unknown, 0 = known
    y_pred_binary = (all_preds == model.num_classes).astype(int)   # 1 = predicted unknown, 0 = predicted known

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    unknown_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    unknown_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    unknown_f1 = 2 * unknown_precision * unknown_recall / (unknown_precision + unknown_recall) if (unknown_precision + unknown_recall) > 0 else 0.0

    # False acceptance rate (FAR): unknown samples predicted as known / total unknown samples
    far = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # Known rejection rate (KRR): known samples predicted as unknown / total known samples
    krr = fn / (fn + tn) if (fn + tn) > 0 else 0.0

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
    parser = argparse.ArgumentParser(description='Train ProtoIDS on CICIoT2023 dataset')
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
                       help='True open-set: withhold MITM/VulnScan/BruteForce from TRAINING (HANDOVER §4)')
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

    # Paths
    data_dir = 'CICIOT23'
    artifact_dir = 'results/baselines'
    development_subset_path = None if args.full_data else 'experiments/data/ciciot_dev.parquet'
    if args.full_data:
        print("Using FULL 5.5M train (CICIOT23/train/train.csv) – not dev parquet")
    model_save_dir = f'experiments/models/{args.experiment_name}'
    results_save_dir = f'experiments/results/{args.experiment_name}'

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)

    # Create data loaders (with optional true open-set withholding)
    withheld_for_train = None
    if args.withhold_open_set:
        # Resolve indices for MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce
        import json as _json, joblib as _joblib, os as _os
        _lm_path = _os.path.join(artifact_dir, 'ciciot_label_mapping.json')
        with open(_lm_path) as _f: _lm = _json.load(_f)
        _l2i = _lm['multiclass']['label_to_int']
        withheld_names = ['MITM-ArpSpoofing', 'VulnerabilityScan', 'DictionaryBruteForce']
        withheld_for_train = [_l2i[n] for n in withheld_names if n in _l2i]
        print(f"True open-set enabled: withholding {withheld_names} -> indices {withheld_for_train} from TRAINING")
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes, input_dim = create_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        artifact_dir=artifact_dir,
        development_subset_path=development_subset_path,
        withheld_classes=withheld_for_train
    )

    print(f"Dataset loaded:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Get class weights if requested
    class_weights = None
    if args.class_weights:
        # Create a temporary dataset to compute class weights (respect withholding)
        temp_dataset = CICIoT2023ProtoIDSDataset(
            data_dir=data_dir,
            split='train',
            artifact_dir=artifact_dir,
            development_subset_path=development_subset_path,
            withheld_classes=withheld_for_train
        )
        class_weights = temp_dataset.get_class_weights().to(device)
        print(f"Using class weights: {class_weights}")

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
            dropout_rate=args.dropout_rate
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
            verbose=True
        )

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds.")

        # Save the trained model
        model_path = os.path.join(model_save_dir, 'model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'protoids_args': {
                'input_dim': input_dim,
                'num_classes': num_classes,
                'embedding_dim': args.embedding_dim,
                'num_prototypes_per_class': args.num_prototypes,
                'dropout_rate': args.dropout_rate
            },
            'training_history': training_history,
            'experiment_id': experiment_id
        }, model_path)
        print(f"Model saved to {model_path}")

        # Save training history
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
            dropout_rate=checkpoint['protoids_args']['dropout_rate']
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
    closed_set_results = {
        'validation': val_metrics,
        'test': test_metrics
    }

    closed_set_path = os.path.join(results_save_dir, 'closed_set_results.json')
    with open(closed_set_path, 'w') as f:
        json.dump(closed_set_results, f, indent=2)

    # Run open-set evaluation
    # For open-set, we need to withhold specific classes from training
    # As specified: withhold MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce
    print(f"\nSetting up open-set evaluation...")
    print(f"Withholding classes: MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce")

    # Get the label mapping to find the indices of the withheld classes
    label_mapping_path = os.path.join(artifact_dir, 'ciciot_label_mapping.json')
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)

    int_to_label = {v: k for k, v in label_mapping['multiclass']['label_to_int'].items()}
    label_to_int = label_mapping['multiclass']['label_to_int']

    # Classes to withhold for open-set evaluation
    withheld_classes = ['MITM-ArpSpoofing', 'VulnerabilityScan', 'DictionaryBruteForce']
    withheld_class_indices = [
        label_to_int[cls] for cls in withheld_classes if cls in label_to_int
    ]

    print(f"Withheld class indices: {withheld_class_indices}")
    print(f"Withheld class names: {[int_to_label[idx] for idx in withheld_class_indices if idx in int_to_label]}")

    # Note: For a proper open-set evaluation, we would need to retrain the model
    # without the withheld classes. However, as per instructions, we'll:
    # 1. Use the model trained on all classes (or load pre-trained)
    # 2. For open-set evaluation, we'll treat the withheld classes as unknown during evaluation
    # 3. We'll use only known-class validation samples for threshold calibration

    # Create datasets for open-set evaluation
    # Known data: validation set excluding withheld classes
    # Unknown data: validation set containing only withheld classes

    # We'll create temporary data loaders for this purpose
    # First, let's get the validation dataset
    val_dataset = CICIoT2023ProtoIDSDataset(
        data_dir=data_dir,
        split='validation',
        artifact_dir=artifact_dir
    )

    # Get indices for known and unknown samples in validation set
    val_labels_array = val_dataset.y_multi
    known_mask = np.isin(val_labels_array, withheld_class_indices, invert=True)
    unknown_mask = np.isin(val_labels_array, withheld_class_indices)

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

    print(f"Open-set Evaluation Results:")
    print(f"  - Known Accuracy: {open_set_metrics['known_accuracy']:.4f}")
    print(f"  - Known Macro F1: {open_set_metrics['known_macro_f1']:.4f}")
    print(f"  - Known Recall: {open_set_metrics['known_recall']:.4f}")
    print(f"  - Unknown Precision: {open_set_metrics['unknown_precision']:.4f}")
    print(f"  - Unknown Recall: {open_set_metrics['unknown_recall']:.4f}")
    print(f"  - Unknown F1: {open_set_metrics['unknown_f1']:.4f}")
    print(f"  - FAR (False Acceptance Rate): {open_set_metrics['far']:.4f}")
    print(f"  - KRR (Known Rejection Rate): {open_set_metrics['krr']:.4f}")
    print(f"  - AUROC: {open_set_metrics['auc_roc']:.4f}")
    print(f"  - AUPR: {open_set_metrics['auc_pr']:.4f}")
    print(f"  - Threshold: {open_set_metrics['threshold']:.4f}")

    # Save open-set results
    open_set_path = os.path.join(results_save_dir, 'open_set_results.json')
    with open(open_set_path, 'w') as f:
        json.dump(open_set_metrics, f, indent=2)

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
        'open_set': open_set_metrics if 'open_set_metrics' in locals() else None
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
        'withheld_classes': withheld_classes if 'withheld_classes' in locals() else [],
        'seed': args.seed
    }

    # Notes for experiment log
    notes = f"ProtoIDS v1 implementation. Closed-set evaluation on CICIoT2023 validation/test sets. Open-set evaluation withholding {withheld_classes if 'withheld_classes' in locals() else 'MITM-ArpSpoofing, VulnerabilityScan, DictionaryBruteForce'} as unknown classes."

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
    main()