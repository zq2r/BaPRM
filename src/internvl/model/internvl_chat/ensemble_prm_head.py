# Copyright (c) OpenMMLab.
# A lightweight ensemble scalar reward head for PRM training.

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleScalarRewardHead(nn.Module):
    """
    Ensemble scalar reward head for process reward modeling.

    This module follows the lightweight-head ensemble idea used in ActivePRM:
    a shared backbone provides hidden states, and multiple independent reward
    heads predict scalar correctness logits at <prm> positions.

    Input:
        hidden_states:
            shape [M, H] or [B, L, H]
            where H is hidden_size.

    Output:
        logits:
            shape [E, M] or [E, B, L]
            where E is num_heads.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.activation = activation

        # Shared normalization before ensemble heads.
        self.norm = nn.LayerNorm(self.hidden_size)

        # Ensemble linear layer 1:
        # each head owns one [hidden_size, hidden_dim] matrix.
        self.w1 = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_size, self.hidden_dim)
        )
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, self.hidden_dim))

        # Ensemble linear layer 2:
        # each head owns one [hidden_dim, 1] matrix.
        self.w2 = nn.Parameter(
            torch.empty(self.num_heads, self.hidden_dim, 1)
        )
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.w1[i])
            nn.init.xavier_uniform_(self.w2[i])
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"but got {hidden_states.size(-1)}"
            )

        # Save original prefix shape: [M] or [B, L].
        original_shape = hidden_states.shape[:-1]

        # Flatten all non-hidden dimensions.
        # [M, H] or [B, L, H] -> [M_total, H]
        x = self.norm(hidden_states)
        x = x.reshape(-1, self.hidden_size)

        # Match dtype with parameters, which is important for bf16/fp16 training.
        x = x.to(dtype=self.w1.dtype)

        # [M, H] x [E, H, D] -> [E, M, D]
        x = torch.einsum("mh,ehd->emd", x, self.w1)
        x = x + self.b1[:, None, :]
        x = self._activate(x)

        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        # [E, M, D] x [E, D, 1] -> [E, M, 1]
        logits = torch.einsum("emd,edo->emo", x, self.w2)
        logits = logits + self.b2[:, None, :]
        logits = logits.squeeze(-1)

        # [E, M_total] -> [E, *original_shape]
        logits = logits.reshape(self.num_heads, *original_shape)
        return logits

    @staticmethod
    def probs_mean_std(logits: torch.Tensor):
        """
        Convert ensemble logits to reward mean and ensemble uncertainty.

        Args:
            logits: [E, ...]
        Returns:
            mean: [ ... ]
            std:  [ ... ]
        """
        probs = torch.sigmoid(logits)
        mean = probs.mean(dim=0)
        std = probs.std(dim=0, unbiased=False)
        return mean, std