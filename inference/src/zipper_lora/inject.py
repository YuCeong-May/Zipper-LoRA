"""PEFT-style model injection and language-context helpers."""

from __future__ import annotations

from collections.abc import Iterable
import torch
from torch import nn

from .config import ZipperLoRAConfig
from .layers import ZipperLoRALinear


DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_zipper_lora(model: nn.Module, config: ZipperLoRAConfig) -> list[str]:
    """Replace selected ``nn.Linear`` children in-place and return their paths.

    Targets may be leaf names (``q_proj``) or full suffixes (``mlp.0``).  If
    ``target_scope`` is set, only paths that start with that scope are wrapped.
    """
    targets = tuple(config.target_modules or DEFAULT_TARGET_MODULES)
    replaced: list[str] = []
    for parent, name, path, child in list(_iter_children(model)):
        if not isinstance(child, nn.Linear) or isinstance(child, ZipperLoRALinear):
            continue
        if config.target_scope and not (path == config.target_scope or path.startswith(config.target_scope + ".")):
            continue
        if any(path == target or path.endswith("." + target) or name == target for target in targets):
            setattr(parent, name, ZipperLoRALinear(child, config))
            replaced.append(path)
    return replaced


def set_zipper_language_ids(model: nn.Module, language_ids: torch.Tensor) -> None:
    """Attach batch language ids to every injected adapter before a forward pass."""
    for module in model.modules():
        if isinstance(module, ZipperLoRALinear):
            module.set_language_ids(language_ids)


def clear_zipper_language_ids(model: nn.Module) -> None:
    """Clear transient routing context after the forward/generation call."""
    for module in model.modules():
        if isinstance(module, ZipperLoRALinear):
            module.clear_language_ids()


def iter_zipper_layers(model: nn.Module) -> Iterable[tuple[str, ZipperLoRALinear]]:
    """Yield injected layers for inspection, checkpointing, or analysis."""
    for name, module in model.named_modules():
        if isinstance(module, ZipperLoRALinear):
            yield name, module


def _iter_children(root: nn.Module):
    stack = [(root, "")]
    while stack:
        parent, prefix = stack.pop()
        for name, child in parent.named_children():
            path = f"{prefix}.{name}" if prefix else name
            yield parent, name, path, child
            stack.append((child, path))
