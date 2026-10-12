"""Inference-only compatibility layers for the released Zipper-LoRA states.

These modules intentionally preserve the parameter names of released
``mp_rank_00_model_states.pt`` files (``base``, ``lora.A``, ``lora.B_shared``
and so on).  They contain no trainer, optimizer, dataset, or loss code.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ZipperRouter(nn.Module):
    def __init__(self, in_features: int = 1280, r: int = 32) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, r)

    def forward(self, embeddings):
        probabilities = torch.sigmoid(self.projection(self.layernorm(embeddings)))
        hard = (probabilities > 0.5).float()
        mask = hard - probabilities.detach() + probabilities if self.training else hard
        return mask, probabilities


class CheckpointZipperLoRA(nn.Module):
    """Rank-wise residual whose state layout matches published checkpoints."""

    def __init__(self, in_features, out_features, shared_embedding, *, num_languages=12,
                 r=32, alpha=None, embedding_dim=1280, static=False, shared_rank=None,
                 soft=False, independent=False):
        super().__init__()
        self.num_languages, self.r = num_languages, r
        self.alpha = alpha or 2.0 * r
        self.static, self.soft, self.independent = static, soft, independent
        self.shared_rank = r // 2 if shared_rank is None else shared_rank
        # A list deliberately prevents duplicate module registration.  The
        # registered copy is attached once to model.audio by the loader.
        self.shared_embedding_container = [shared_embedding]
        if independent:
            self.A_specific = nn.Parameter(torch.empty(num_languages, r, in_features))
            self.B_specific = nn.Parameter(torch.empty(num_languages, out_features, r))
            self.router = self.A = self.B_shared = None
        elif static:
            self.register_buffer("A", torch.empty(r, in_features))
            self.B_shared = nn.Parameter(torch.empty(out_features, self.shared_rank))
            self.B_specific = nn.Parameter(torch.empty(num_languages, out_features, r - self.shared_rank))
            self.router = None
        else:
            self.register_buffer("A", torch.empty(r, in_features))
            self.B_shared = nn.Parameter(torch.empty(out_features, r))
            self.B_specific = nn.Parameter(torch.empty(num_languages, out_features, r))
            self.router = ZipperRouter(embedding_dim, r)
        self.register_buffer("lid_ids", None, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.independent:
            nn.init.normal_(self.A_specific, std=0.02)
        else:
            nn.init.normal_(self.A, std=0.02)
            nn.init.zeros_(self.B_shared)
        nn.init.zeros_(self.B_specific)

    def set_lid_ids(self, ids):
        self.lid_ids = ids

    def clear_lid_ids(self):
        self.lid_ids = None

    def _ids(self, x):
        if self.lid_ids is None:
            raise RuntimeError("Set Zipper-LoRA language IDs before inference.")
        ids = self.lid_ids.to(x.device, dtype=torch.long).flatten()
        if ids.numel() == 1:
            ids = ids.expand(x.shape[0])
        if ids.numel() != x.shape[0]:
            raise ValueError(f"Expected one or {x.shape[0]} language IDs, got {ids.numel()}.")
        return ids.clamp(0, self.num_languages - 1)

    @staticmethod
    def _batched(x, weights):
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weights.transpose(1, 2)).squeeze(1)
        return torch.bmm(x, weights.transpose(1, 2))

    def forward(self, x):
        dtype = x.dtype
        ids = self._ids(x)
        if self.independent:
            a = self.A_specific.index_select(0, ids).to(dtype)
            b = self.B_specific.index_select(0, ids).to(dtype)
            ranks = torch.bmm(
                x.unsqueeze(1) if x.ndim == 2 else x, a.transpose(1, 2)
            )
            if x.ndim == 2:
                ranks = ranks.squeeze(1)
            return self._batched(ranks, b) * (self.alpha / self.r)

        ranks = F.linear(x, self.A.to(dtype))
        if self.static:
            result = F.linear(ranks[..., :self.shared_rank], self.B_shared.to(dtype)) if self.shared_rank else 0.0
            if self.shared_rank < self.r:
                result = result + self._batched(ranks[..., self.shared_rank:], self.B_specific.index_select(0, ids).to(dtype))
            return result * (self.alpha / self.r)

        embedding = self.shared_embedding_container[0].to(x.device)(ids)
        mask, probabilities = self.router(embedding)
        mask = (probabilities if self.soft else mask).to(dtype)
        mask = mask.unsqueeze(1) if x.ndim == 3 else mask
        shared = F.linear(ranks * mask, self.B_shared.to(dtype))
        specific = self._batched(ranks * (1.0 - mask), self.B_specific.index_select(0, ids).to(dtype))
        return (shared + specific) * (self.alpha / self.r)


class CheckpointZipperAdapter(nn.Module):
    """``nn.Linear`` wrapper retaining the published ``base``/``lora`` keys."""
    def __init__(self, base: nn.Linear, **kwargs):
        super().__init__()
        self.base = base
        self.lora = CheckpointZipperLoRA(base.in_features, base.out_features, **kwargs)

    def forward(self, x):
        return self.base(x) + self.lora(x)

    def set_lid_ids(self, ids):
        self.lora.set_lid_ids(ids)

    def clear_lid_ids(self):
        self.lora.clear_lid_ids()


class CheckpointFlyLoRA(nn.Module):
    """FlyLoRA residual: frozen sparse ``A``, top-k rank activation, ``B`` up-projection."""

    def __init__(self, in_features, out_features, *, r=32, k=8, alpha=None):
        super().__init__()
        self.r, self.k = r, k
        self.alpha = alpha or 2.0 * r
        self.register_buffer("A", torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        # Load-balancing bias: only selects ranks, never added to the output.
        self.register_buffer("d", torch.zeros(r))

    def forward(self, x):
        y = F.linear(x, self.A.to(x.dtype))
        _, selected = torch.topk((y + self.d.to(x.dtype)).abs(), self.k, dim=-1)
        mask = torch.zeros_like(y).scatter_(-1, selected, 1.0)
        return F.linear(y * mask, self.B.to(x.dtype)) * (self.alpha / self.r)


class CheckpointFlyLoRAAdapter(nn.Module):
    """``nn.Linear`` wrapper retaining the published ``base``/``fly`` keys."""
    def __init__(self, base: nn.Linear, **kwargs):
        super().__init__()
        self.base = base
        self.fly = CheckpointFlyLoRA(base.in_features, base.out_features, **kwargs)

    def forward(self, x):
        return self.base(x) + self.fly(x)


AUDIO_TARGETS = ("query", "value", "key", "out", "mlp.0", "mlp.2")


def _replace_audio_linears(model, factory):
    stack = [(model, "")]
    while stack:
        parent, prefix = stack.pop()
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            is_audio = "model.audio" in path or path.startswith("audio") or ".audio." in path
            match = name in AUDIO_TARGETS or any(path.endswith("." + target) for target in AUDIO_TARGETS)
            if isinstance(child, nn.Linear) and is_audio and match:
                setattr(parent, name, factory(child))
            else:
                stack.append((child, path))
    return model


def inject_checkpoint_zipper_lora(model, shared_embedding, *, variant="soft", num_languages=12,
                                 r=32, static_shared_rank=None):
    """Inject inference adapters into the released Whisper encoder layout."""
    return _replace_audio_linears(model, lambda child: CheckpointZipperAdapter(
        child, shared_embedding=shared_embedding, num_languages=num_languages, r=r,
        static=variant == "static", soft=variant == "soft",
        independent=variant == "independent", shared_rank=static_shared_rank,
    ))


def inject_checkpoint_flylora(model, *, r=32, k=8):
    """Inject FlyLoRA adapters into the released Whisper encoder layout."""
    return _replace_audio_linears(model, lambda child: CheckpointFlyLoRAAdapter(child, r=r, k=k))


def remap_static_moe_state(state):
    """Rename released Static states saved by the LID-routed rank-MoE trainer.

    ``moe.{A,B_shared,B_experts}`` become ``lora.{A,B_shared,B_specific}``: with
    LID routing the expert bank is indexed by language exactly like Static
    Zipper-LoRA.  ``chunk_moe_router`` is bypassed by LID routing and dropped.
    """
    if not any(".moe.B_experts" in key for key in state):
        return state
    return {key.replace(".moe.B_experts", ".lora.B_specific").replace(".moe.", ".lora."): value
            for key, value in state.items() if ".chunk_moe_router." not in key}


def set_checkpoint_language_ids(model, ids):
    for module in model.modules():
        if isinstance(module, CheckpointZipperAdapter):
            module.set_lid_ids(ids)


def clear_checkpoint_language_ids(model):
    for module in model.modules():
        if isinstance(module, CheckpointZipperAdapter):
            module.clear_lid_ids()
