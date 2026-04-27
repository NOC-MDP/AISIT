"""
model.py — Residual MLP for conservative tracer prediction.

Architecture
------------
Input  → Linear → [ResidualBlock × N] → Linear → scalar output

Each ResidualBlock:
    x → LayerNorm → Linear → Activation → Dropout → Linear → + x

The residual connections stabilise training on geophysical data where
the relationship between tracers and hydrographic variables can be
non-linear but smooth.
"""

import torch
import torch.nn as nn
from typing import List

from config import ModelConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _get_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }[name]


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block with LayerNorm.

    Uses a projection layer when in_dim ≠ out_dim so the skip connection
    always has matching dimensions.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str, dropout: float):
        super().__init__()
        self.norm    = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act     = _get_activation(activation)
        self.drop    = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)

        # Projection for skip connection when dimensions differ
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        # Small initialisation on output linear to start near identity mapping
        nn.init.zeros_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        return out + residual


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TracerMLP(nn.Module):
    """
    Residual MLP that maps co-located gridded ocean variables → tracer value.

    Parameters
    ----------
    n_features  : number of input features (set automatically from data)
    hidden_dims : list of hidden layer widths, e.g. [128, 256, 256, 128]
    dropout     : dropout probability (applied inside each residual block)
    activation  : activation function name
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dims[0]),
            _get_activation(activation),
        )

        # Residual blocks
        dims = hidden_dims
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1], activation, dropout)
            for i in range(len(dims) - 1)
        ])

        # Output head
        self.output_head = nn.Linear(dims[-1], 1)

        self._n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, cfg: ModelConfig, n_features: int) -> "TracerMLP":
        return cls(
            n_features  = n_features,
            hidden_dims = cfg.hidden_dims,
            dropout     = cfg.dropout,
            activation  = cfg.activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, n_features)  normalised input features

        Returns
        -------
        out : (batch, 1)  predicted normalised tracer value
        """
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return self.output_head(out)

    def predict_numpy(
        self,
        X: "np.ndarray",
        device: torch.device,
        batch_size: int = 4096,
    ) -> "np.ndarray":
        """
        Convenience method: numpy in → numpy out, handles batching internally.
        Used during full-grid inference to avoid OOM errors.
        """
        import numpy as np
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_b = torch.from_numpy(X[i: i + batch_size]).to(device)
                preds.append(self(x_b).squeeze(-1).cpu().numpy())
        return np.concatenate(preds)

    def __repr__(self):
        return (
            f"TracerMLP("
            f"params={self._n_params:,}, "
            f"blocks={len(self.blocks)}"
            f")"
        )
