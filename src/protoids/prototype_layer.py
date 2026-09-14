"""
ProtoIDS Prototype Layer
Learns multiple prototypes per class in the embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class PrototypeLayer(nn.Module):
    """
    Prototype layer that learns multiple prototypes per class.
    Prototypes are initialized using K-means and then learnable parameters.
    """

    def __init__(self, num_classes, embedding_dim, num_prototypes_per_class=3):
        """
        Initialize the prototype layer.

        Args:
            num_classes: Number of classes
            embedding_dim: Dimension of the embedding space
            num_prototypes_per_class: Number of prototypes per class (default: 3)
        """
        super(PrototypeLayer, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes_total = num_classes * num_prototypes_per_class

        # Learnable prototype parameters
        self.prototypes = nn.Parameter(
            torch.Tensor(self.num_prototypes_total, embedding_dim)
        )

        # Initialize prototypes
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize prototypes using K-means (will be updated during training)."""
        # Initialize with small random values
        nn.init.normal_(self.prototypes, mean=0.0, std=0.1)

    def initialize_prototypes(self, embeddings, labels):
        """
        Initialize prototypes using K-means clustering on embeddings.

        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: Tensor of shape (num_samples,) with class labels
        """
        self.eval()
        with torch.no_grad():
            embeddings_np = embeddings.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for class_idx in range(self.num_classes):
                # Get embeddings for this class
                class_mask = labels_np == class_idx
                if np.sum(class_mask) == 0:
                    # If no samples for this class, skip initialization
                    continue

                class_embeddings = embeddings_np[class_mask]

                # Apply K-means to get initial prototypes
                kmeans = KMeans(
                    n_clusters=self.num_prototypes_per_class,
                    random_state=42,
                    n_init=10
                )
                kmeans.fit(class_embeddings)

                # Store the cluster centers as initial prototypes
                start_idx = class_idx * self.num_prototypes_per_class
                end_idx = start_idx + self.num_prototypes_per_class
                self.prototypes.data[start_idx:end_idx] = torch.from_numpy(
                    kmeans.cluster_centers_
                ).float()

    def forward(self, embeddings):
        """
        Compute distances from embeddings to prototypes.

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) (L2 normalized)

        Returns:
            distances: Tensor of shape (batch_size, num_prototypes_total)
                      where distances[i, j] is the cosine distance between
                      embedding i and prototype j
            min_distances_per_class: Tensor of shape (batch_size, num_classes)
                                   where min_distances_per_class[i, c] is the
                                   minimum distance from embedding i to any prototype of class c
        """
        # Compute cosine distance: d(z, p) = 1 - cos(z, p)
        # Since embeddings are L2 normalized, cos(z, p) = z · p
        # So d(z, p) = 1 - z · p

        # Normalize prototypes to unit length for cosine distance
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Compute cosine similarity: (batch_size, embedding_dim) @ (num_prototypes, embedding_dim)^T
        # = (batch_size, num_prototypes)
        cosine_similarity = torch.matmul(embeddings, prototypes_norm.t())

        # Convert to cosine distance: d(z, p) = 1 - cos(z, p)
        distances = 1.0 - cosine_similarity  # Shape: (batch_size, num_prototypes_total)

        # Reshape to (batch_size, num_classes, num_prototypes_per_class)
        distances_reshaped = distances.view(
            -1, self.num_classes, self.num_prototypes_per_class
        )

        # Find minimum distance to prototypes of each class
        min_distances_per_class = torch.min(
            distances_reshaped, dim=2
        )[0]  # Shape: (batch_size, num_classes)

        return distances, min_distances_per_class