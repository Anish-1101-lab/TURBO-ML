"""
Probe architectures. Both output raw logits (sigmoid applied at loss/eval
time via BCEWithLogitsLoss / torch.sigmoid) so the two probe types share the
same training and evaluation code paths.
"""
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    """Small single-hidden-layer MLP -- the nonlinearity check against the
    linear probe, not meant to be a large model relative to the data."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
