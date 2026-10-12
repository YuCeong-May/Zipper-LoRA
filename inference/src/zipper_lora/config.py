"""Public configuration for the inference/model-composition package."""

from dataclasses import dataclass
from typing import Literal, Sequence


ZipperVariant = Literal["static", "hard", "soft"]


@dataclass
class ZipperLoRAConfig:
    """Describes one Zipper-LoRA adapter.

    ``variant`` selects Static, Hard, or Soft rank-wise construction.  This
    package intentionally contains no optimizer, loss, dataset, or trainer
    configuration.
    """

    num_languages: int
    lid_embedding_dim: int
    r: int = 32
    alpha: float | None = None
    variant: ZipperVariant = "soft"
    shared_rank: int | None = None
    threshold: float = 0.5
    target_modules: Sequence[str] | None = None
    target_scope: str | None = None
    freeze_base_model: bool = True

    def __post_init__(self) -> None:
        if self.num_languages < 1 or self.lid_embedding_dim < 1 or self.r < 1:
            raise ValueError("num_languages, lid_embedding_dim, and r must be positive.")
        if self.variant not in {"static", "hard", "soft"}:
            raise ValueError("variant must be 'static', 'hard', or 'soft'.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        if self.shared_rank is not None and not 0 <= self.shared_rank <= self.r:
            raise ValueError("shared_rank must lie in [0, r].")

    @property
    def scaling(self) -> float:
        return (self.alpha if self.alpha is not None else 2.0 * self.r) / self.r

    @property
    def resolved_shared_rank(self) -> int:
        return self.r // 2 if self.shared_rank is None else self.shared_rank
