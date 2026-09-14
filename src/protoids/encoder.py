"""
ProtoIDS Encoder Module
Maps input features to a normalized embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoIDSEncoder(nn.Module):
    """
    Encoder network that maps input features to a normalized embedding space.

    Architecture:
    Input → Linear(128) → BatchNorm → ReLU → Dropout(0.2) → Linear(64) → ReLU → Linear(32) → L2 normalization
    """

    def __init__(self, input_dim, embedding_dim=32, dropout_rate=0.2):
        """
        Initialize the encoder.

        Args:
            input_dim: Number of input features
            embedding_dim: Dimension of the embedding space (default: 32)
            dropout_rate: Dropout rate (default: 0.2)
        """
        super(ProtoIDSEncoder, self).__init__()

        self.embedding_dim = embedding_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

            input: Tensor of shape (batch_size, input_dim)
            return: Tuple of (embedding, normalized_embedding)
                    embedding: Raw embedding before normalization
                    normalized_embedding: L2 normalized embedding
        """
        # First layer: Linear -> BatchNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second layer: Linear -> ReLU
        x = self.fc2(x)
        x = F.relu(x)

        # Third layer: Linear (to embedding space)
        embedding = self.fc3(x)

        # L2 normalization
        normalized_embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, normalized_embedding