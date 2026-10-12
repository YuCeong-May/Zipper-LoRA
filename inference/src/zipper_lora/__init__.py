"""A minimal, training-agnostic Zipper-LoRA package."""

from .config import ZipperLoRAConfig
from .inject import (
    DEFAULT_TARGET_MODULES,
    clear_zipper_language_ids,
    inject_zipper_lora,
    iter_zipper_layers,
    set_zipper_language_ids,
)
from .layers import LIDRouter, ZipperLoRA, ZipperLoRALinear
from .checkpoint import (
    CheckpointZipperAdapter,
    CheckpointZipperLoRA,
    clear_checkpoint_language_ids,
    inject_checkpoint_zipper_lora,
    set_checkpoint_language_ids,
)

__all__ = [
    "DEFAULT_TARGET_MODULES", "LIDRouter", "ZipperLoRA", "ZipperLoRAConfig",
    "ZipperLoRALinear", "clear_zipper_language_ids", "inject_zipper_lora",
    "iter_zipper_layers", "set_zipper_language_ids",
    "CheckpointZipperAdapter", "CheckpointZipperLoRA", "clear_checkpoint_language_ids",
    "inject_checkpoint_zipper_lora", "set_checkpoint_language_ids",
]
