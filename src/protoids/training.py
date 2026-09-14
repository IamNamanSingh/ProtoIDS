"""
ProtoIDS Training Module
Implements the training loop with classification and compactness losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
import os
from typing import Dict, Tuple, Optional
from .protoids_model import ProtoIDS


def compactness_loss(embeddings, labels, prototype_layer, num_classes):
    """
    Compute compactness loss: encourage embeddings to be close to their class's nearest prototype.

    Args:
        embeddings: Normalized embeddings (batch_size, embedding_dim)
        labels: Ground truth labels (batch_size,)
        prototype_layer: PrototypeLayer instance
        num_classes: Number of classes

    Returns:
        loss: Compactness loss scalar
    """
    # Get minimum distances to prototypes of each class
    _, min_distances_per_class = prototype_layer(embeddings)  # (batch_size, num_classes)

    # For each sample, get the distance to its true class's nearest prototype
    batch_size = embeddings.size(0)
    true_class_distances = min_distances_per_class[
        torch.arange(batch_size), labels
    ]  # Shape: (batch_size,)

    # Compactness loss is the mean of these distances
    loss = torch.mean(true_class_distances)

    return loss


def classification_loss(distances_per_class, labels, weighting=None):
    """
    Compute classification loss based on distances to class prototypes.

    We convert distances to similarities and use cross-entropy loss.
    Smaller distance -> larger similarity -> higher probability.

    Args:
        distances_per_class: Minimum distances to prototypes of each class (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        weighting: Optional class weights for weighted loss

    Returns:
        loss: Classification loss scalar
    """
    # Convert distances to similarities (using negative distance so smaller distance = larger similarity)
    # We add a small epsilon to avoid numerical issues
    similarities = -distances_per_class  # Shape: (batch_size, num_classes)

    # Cross-entropy loss expects raw logits, which similarities can serve as
    loss_fn = nn.CrossEntropyLoss(weight=weighting)
    loss = loss_fn(similarities, labels)

    return loss


def train_protoids(model, train_loader, val_loader, device, num_epochs=100,
                   learning_rate=0.001, lambda_compact=0.1,
                   class_weights=None, verbose=True):
    """
    Train the ProtoIDS model.

    Args:
        model: ProtoIDS model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on ('cpu' or 'cuda')
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        lambda_compact: Weight for compactness loss (default: 0.1)
        class_weights: Optional tensor of class weights for weighted loss
        verbose: Whether to print training progress

    Returns:
        training_history: Dictionary containing training and validation metrics
    """
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Tracking variables
    training_history = {
        'train_loss': [],
        'train_class_loss': [],
        'train_compact_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_macro_f1': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_compact_loss = 0.0
        num_train_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            embedding, normalized_embedding, distances, min_distances_per_class = model(batch_X)

            # Compute losses
            class_loss = classification_loss(min_distances_per_class, batch_y, class_weights)
            compact_loss = compactness_loss(normalized_embedding, batch_y,
                                            model.prototype_layer, model.num_classes)
            loss = class_loss + lambda_compact * compact_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate losses
            train_loss += loss.item()
            train_class_loss += class_loss.item()
            train_compact_loss += compact_loss.item()
            num_train_batches += 1

        # Average training losses
        avg_train_loss = train_loss / num_train_batches
        avg_train_class_loss = train_class_loss / num_train_batches
        avg_train_compact_loss = train_compact_loss / num_train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_class_loss = 0.0
        val_compact_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        num_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                embedding, normalized_embedding, distances, min_distances_per_class = model(batch_X)

                # Compute losses
                class_loss = classification_loss(min_distances_per_class, batch_y, class_weights)
                compact_loss = compactness_loss(normalized_embedding, batch_y,
                                                model.prototype_layer, model.num_classes)
                loss = class_loss + lambda_compact * compact_loss

                # Accumulate losses
                val_loss += loss.item()
                val_class_loss += class_loss.item()
                val_compact_loss += compact_loss.item()
                num_val_batches += 1

                # Get predictions for metrics
                predicted_classes, _ = model.predict_class(normalized_embedding)
                all_val_preds.append(predicted_classes.cpu())
                all_val_labels.append(batch_y.cpu())

        # Average validation losses
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_class_loss = val_class_loss / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_compact_loss = val_compact_loss / num_val_batches if num_val_batches > 0 else 0.0

        # Compute validation metrics
        if len(all_val_preds) > 0:
            val_preds = torch.cat(all_val_preds)
            val_labels = torch.cat(all_val_labels)
            val_accuracy = (val_preds == val_labels).float().mean().item()

            # Compute macro F1
            from sklearn.metrics import f1_score
            val_macro_f1 = f1_score(
                val_labels.numpy(), val_preds.numpy(),
                average='macro', zero_division=0
            )
        else:
            val_accuracy = 0.0
            val_macro_f1 = 0.0

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_class_loss'].append(avg_train_class_loss)
        training_history['train_compact_loss'].append(avg_train_compact_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        training_history['val_macro_f1'].append(val_macro_f1)
        training_history['learning_rates'].append(current_lr)

        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f'Epoch {epoch:3d} | '
                  f'Train Loss: {avg_train_loss:.4f} (Class: {avg_train_class_loss:.4f}, '
                  f'Compact: {avg_train_compact_loss:.4f}) | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Val Acc: {val_accuracy:.4f} | '
                  f'Val Macro F1: {val_macro_f1:.4f} | '
                  f'LR: {current_lr:.6f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Note: Model saving is handled in the main training script

    return training_history