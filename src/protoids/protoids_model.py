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
                 num_prototypes_per_class=3, dropout_rate=0.2):
        """
        Initialize the ProtoIDS model.

        Args:
            input_dim: Number of input features
            num_classes: Number of classes
            embedding_dim: Dimension of the embedding space (default: 32)
            num_prototypes_per_class: Number of prototypes per class (default: 3)
            dropout_rate: Dropout rate in encoder (default: 0.2)
        """
        super(ProtoIDS, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_prototypes_per_class = num_prototypes_per_class

        # Encoder
        self.encoder = ProtoIDSEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate
        )

        # Prototype layer
        self.prototype_layer = PrototypeLayer(
            num_classes=num_classes,
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
        """
        # Get embeddings from encoder
        embedding, normalized_embedding = self.encoder(x)

        # Compute distances to prototypes
        distances, min_distances_per_class = self.prototype_layer(normalized_embedding)

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
        predicted_classes = torch.argmin(min_distances_per_class, dim=1)
        min_distances = torch.min(min_distances_per_class, dim=1)[0]
        return predicted_classes, min_distances

    def predict_unknown(self, normalized_embedding, threshold):
        """
        Predict class or UNKNOWN based on distance threshold.

        Args:
            normalized_embedding: L2 normalized embeddings (batch_size, embedding_dim)
            threshold: Distance threshold for UNKNOWN rejection

        Returns:
            predictions: Tensor of shape (batch_size,) with predicted class indices
                        (num_classes indicates UNKNOWN)
            min_distances: Tensor of shape (batch_size,) with minimum distances
        """
        predicted_classes, min_distances = self.predict_class(normalized_embedding)

        # If minimum distance exceeds threshold, predict UNKNOWN (represented as num_classes)
        unknown_mask = min_distances > threshold
        predictions = predicted_classes.clone()
        predictions[unknown_mask] = self.num_classes  # Use num_classes as UNKNOWN class index

        return predictions, min_distances