"""
ProtoIDS Model
Combines encoder and prototype layers for protoyp
-based classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ProtoIDSEncoder
from .prototype_layer import PrototypeLayer


class ProtoIDS(nn.Module):
    """
    ProtoIDS model for prototype-based intrusion detection.

    Architecture:
    Input → Encoder → Embedding → Prototype Layer → Distances → Classification
    """

    def __init__(self, input_dim, num_classes, embedding_dim=32,
                 num_prototypes_per_class=3, dropout_rate=0.2,
                 unknown_class_index=None):
        """
        Initialize the ProtoIDS model.

        Args:
            input_dim: Number of input features
            num_classes: Number of classes (including unknown class if applicable)
            embedding_dim: Dimension of the embedding space (default: 32)
            num_prototypes_per_class: Number of prototypes per class (default: 3)
            dropout_rate: Dropout rate in encoder (default: 0.2)
            unknown_class_index: Index of the unknown class (if applicable), else None
        """
        super(ProtoIDS, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes  # total classes (including unknown if applicable)
        self.embedding_dim = embedding_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.unknown_class_index = unknown_class_index
        self.dropout_rate = dropout_rate

        # Determine number of classes that have prototypes (known classes)
        if unknown_class_index is not None:
            # Assume unknown class is a single class; we will not create prototypes for it
            self.num_known_classes = num_classes - 1
        else:
            self.num_known_classes = num_classes

        # Encoder
        self.encoder = ProtoIDSEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        # Prototype layer - only for known classes
        self.prototype_layer = PrototypeLayer(
            num_classes=self.num_known_classes,
            embedding_dim=embedding_dim,
            num_prototypes_per_class=num_prototypes_per_class
        )

    def forward(self, x):
        """
        Forward pass through the ProtoIDS model.

        Args:
            x: Tensor of shape (batch_size, input_dim)

        Returns:
            embedding: Raw embedding before normalization (batch_size, embedding_dim)
            normalized_embedding: L2 normalized embedding (batch_size, embedding_dim)
            distances: Cosine distances to all prototypes (batch_size, num_prototypes_total)
            min_distances_per_class: Minimum distance to prototypes of each class (batch_size, num_classes)
                                   For the unknown class, distance is set to a large value (inf) so it is never selected.
        """
        # Get embeddings from encoder
        embedding, normalized_embedding = self.encoder(x)

        # Compute distances to prototypes (only for known classes)
        distances, min_distances_per_class_known = self.prototype_layer(normalized_embedding)
        # distances: (batch, num_known_classes * num_prototypes_per_class)
        # min_distances_per_class_known: (batch, num_known_classes)

        # Expand to include unknown class with large distance
        if self.unknown_class_index is not None:
            # Create a tensor for unknown class distances: large value (e.g., 1e6)
            unknown_distances = torch.full(
                (min_distances_per_class_known.size(0), 1),
                fill_value=1e6,
                device=min_distances_per_class_known.device,
                dtype=min_distances_per_class_known.dtype
            )
            # Insert unknown class distance at the appropriate position
            # We need to build min_distances_per_class of shape (batch, num_classes)
            # where unknown_class_index gets the large distance.
            # We'll construct by concatenating slices.
            if self.unknown_class_index == 0:
                min_distances_per_class = torch.cat([unknown_distances, min_distances_per_class_known], dim=1)
            elif self.unknown_class_index == self.num_classes:
                min_distances_per_class = torch.cat([min_distances_per_class_known, unknown_distances], dim=1)
            else:
                left = min_distances_per_class_known[:, :self.unknown_class_index]
                right = min_distances_per_class_known[:, self.unknown_class_index:]
                min_distances_per_class = torch.cat([left, unknown_distances, right], dim=1)
        else:
            min_distances_per_class = min_distances_per_class_known

        return embedding, normalized_embedding, distances, min_distances_per_class

    def predict_class(self, normalized_embedding):
        """
        Predict class based on minimum distance to class prototypes.

        Args:
            normalized_embedding: L2 normalized embeddings (batch_size, embedding_dim)

        Returns:
            predicted_classes: Tensor of shape (batch_size,) with predicted class indices
            min_distances: Tensor of shape (batch_size,) with minimum distances
        """
        _, min_distances_per_class = self.prototype_layer(normalized_embedding)
        # Expand to include unknown class with large distance as in forward
        if self.unknown_class_index is not None:
            unknown_distances = torch.full(
                (min_distances_per_class.size(0), 1),
                fill_value=1e6,
                device=min_distances_per_class.device,
                dtype=min_distances_per_class.dtype
            )
            if self.unknown_class_index == 0:
                min_distances_per_class_exp = torch.cat([unknown_distances, min_distances_per_class], dim=1)
            elif self.unknown_class_index == self.num_classes:
                min_distances_per_class_exp = torch.cat([min_distances_per_class, unknown_distances], dim=1)
            else:
                left = min_distances_per_class[:, :self.unknown_class_index]
                right = min_distances_per_class[:, self.unknown_class_index:]
                min_distances_per_class_exp = torch.cat([left, unknown_distances, right], dim=1)
        else:
            min_distances_per_class_exp = min_distances_per_class

        predicted_classes = torch.argmin(min_distances_per_class_exp, dim=1)
        min_distances = torch.min(min_distances_per_class_exp, dim=1)[0]
        return predicted_classes, min_distances

    def predict_unknown(self, normalized_embedding, threshold):
        """
        Predict class or UNKNOWN based on distance threshold.

        Args:
            normalized_embedding: L2 normalized embeddings (batch_size, embedding_dim)
            threshold: Distance threshold for UNKNOWN rejection

        Returns:
            predictions: Tensor of shape (batch_size,) with predicted class indices
                        (unknown_class_index indicates UNKNOWN)
            min_distances: Tensor of shape (batch_size,) with minimum distances
        """
        predicted_classes, min_distances = self.predict_class(normalized_embedding)

        # If minimum distance exceeds threshold, predict UNKNOWN (represented as unknown_class_index)
        unknown_mask = min_distances > threshold
        predictions = predicted_classes.clone()
        predictions[unknown_mask] = self.unknown_class_index  # Use unknown_class_index as UNKNOWN class index

        return predictions, min_distances