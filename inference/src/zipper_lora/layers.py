"""Zipper-LoRA layer definitions: Static, Hard, and Soft composition."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ZipperLoRAConfig


class LIDRouter(nn.Module):
    """LayerNorm -> linear -> sigmoid language-conditioned rank router.

    The returned value is a *shared* gate.  It is the complement of the
    language-specific mixing weight used in the paper's Soft equation.
    """

    def __init__(self, embedding_dim: int, rank: int) -> None:
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, rank)

    def forward(self, lid_embeddings: Tensor) -> Tensor:
        return torch.sigmoid(self.projection(self.normalizer(lid_embeddings)))


class ZipperLoRA(nn.Module):
    """Low-rank residual with language-conditioned rank-wise B construction."""

    def __init__(self, in_features: int, out_features: int, config: ZipperLoRAConfig) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(torch.empty(config.r, in_features))
        self.lid_embedding = nn.Embedding(config.num_languages, config.lid_embedding_dim)
        self.router = None if config.variant == "static" else LIDRouter(config.lid_embedding_dim, config.r)
        if config.variant == "static":
            shared_rank = config.resolved_shared_rank
            self.B_shared = nn.Parameter(torch.empty(out_features, shared_rank))
            self.B_specific = nn.Parameter(torch.empty(config.num_languages, out_features, config.r - shared_rank))
        else:
            self.B_shared = nn.Parameter(torch.empty(out_features, config.r))
            self.B_specific = nn.Parameter(torch.empty(config.num_languages, out_features, config.r))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B_shared)
        nn.init.zeros_(self.B_specific)
        nn.init.normal_(self.lid_embedding.weight, mean=0.0, std=0.02)

    def routing_weights(self, language_ids: Tensor) -> Tensor | None:
        """Return shared gates of shape ``[batch, r]``; Static has no router."""
        if self.router is None:
            return None
        return self.router(self.lid_embedding(language_ids.long()))

    def forward(self, x: Tensor, language_ids: Tensor) -> Tensor:
        language_ids = _batch_language_ids(language_ids, x.shape[0], x.device)
        ranks = F.linear(x, self.A.to(dtype=x.dtype))
        specific = self.B_specific.index_select(0, language_ids).to(dtype=x.dtype)

        if self.config.variant == "static":
            shared_r = self.config.resolved_shared_rank
            output = F.linear(ranks[..., :shared_r], self.B_shared.to(dtype=x.dtype))
            if shared_r < self.config.r:
                output = output + _per_sample_linear(ranks[..., shared_r:], specific)
        else:
            shared_gate = self.routing_weights(language_ids).to(dtype=x.dtype)
            if self.config.variant == "hard":
                # Straight-through binary gate: discrete forward, sigmoid gradient.
                discrete = (shared_gate >= self.config.threshold).to(x.dtype)
                if self.training:
                    shared_gate = discrete - shared_gate.detach() + shared_gate
                else:
                    shared_gate = discrete
            shared = F.linear(ranks * shared_gate.unsqueeze(1) if ranks.ndim == 3 else ranks * shared_gate,
                              self.B_shared.to(dtype=x.dtype))
            private_ranks = ranks * (1.0 - shared_gate.unsqueeze(1) if ranks.ndim == 3 else 1.0 - shared_gate)
            output = shared + _per_sample_linear(private_ranks, specific)
        return output * self.config.scaling


class ZipperLoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with a Zipper-LoRA residual."""

    def __init__(self, base_layer: nn.Linear, config: ZipperLoRAConfig) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.zipper_lora = ZipperLoRA(base_layer.in_features, base_layer.out_features, config)
        self._language_ids: Tensor | None = None
        if config.freeze_base_model:
            for parameter in self.base_layer.parameters():
                parameter.requires_grad_(False)

    def set_language_ids(self, language_ids: Tensor) -> None:
        self._language_ids = language_ids

    def clear_language_ids(self) -> None:
        self._language_ids = None

    def forward(self, x: Tensor) -> Tensor:
        if self._language_ids is None:
            raise RuntimeError("Zipper-LoRA needs language_ids. Call set_zipper_language_ids(model, ids) before forward.")
        return self.base_layer(x) + self.zipper_lora(x, self._language_ids)


def _batch_language_ids(language_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
    language_ids = language_ids.to(device=device, dtype=torch.long).flatten()
    if language_ids.numel() == 1:
        return language_ids.expand(batch_size)
    if language_ids.numel() != batch_size:
        raise ValueError(f"Expected one language id or {batch_size} ids, got {language_ids.numel()}.")
    return language_ids


def _per_sample_linear(x: Tensor, weights: Tensor) -> Tensor:
    """Apply a separately selected ``[out, rank]`` matrix to each batch row."""
    if x.ndim == 2:
        return torch.bmm(x.unsqueeze(1), weights.transpose(1, 2)).squeeze(1)
    if x.ndim == 3:
        return torch.bmm(x, weights.transpose(1, 2))
    raise ValueError("Zipper-LoRA supports [batch, hidden] or [batch, sequence, hidden] tensors.")
