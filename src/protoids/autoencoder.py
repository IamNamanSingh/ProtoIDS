"""Autoencoder for ProtoIDS pretraining: 41->128->32->128->41"""
import torch, torch.nn as nn, torch.nn.functional as F
from .encoder import ProtoIDSEncoder

class ProtoIDSAutoEncoder(nn.Module):
    def __init__(self, input_dim=41, embedding_dim=32, dropout_rate=0.2):
        super().__init__()
        self.encoder = ProtoIDSEncoder(input_dim, embedding_dim, dropout_rate)
        # Decoder mirrors encoder: 32 -> 64 -> 128 -> 41
        self.decoder_fc1 = nn.Linear(embedding_dim, 64)
        self.decoder_bn1 = nn.BatchNorm1d(64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_bn2 = nn.BatchNorm1d(128)
        self.decoder_fc3 = nn.Linear(128, input_dim)

    def forward(self, x):
        emb, norm = self.encoder(x)  # norm is L2, but decoder uses raw emb for reconstruction
        h = F.relu(self.decoder_bn1(self.decoder_fc1(emb)))
        h = F.relu(self.decoder_bn2(self.decoder_fc2(h)))
        recon = self.decoder_fc3(h)
        return recon, emb, norm
