"""Order Encoder for TowerController."""
import torch
import torch.nn as nn


class OrderEncoder(nn.Module):
    """Encode variable-length order sets."""

    def __init__(self, order_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(order_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, x):
        """x: (batch, N, order_dim) or (N, order_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.proj(x)
        x = self.encoder(x)
        return x  # (batch, N, hidden_dim)
