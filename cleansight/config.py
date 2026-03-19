"""Configuration for CleanSight defense."""

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class CleanSightConfig:
    """Configuration for the CleanSight test-time backdoor defense.

    Attributes:
        detection_layers: Layer indices used for computing the vision-to-text
            attention ratio feature vector.  Paper default: layers 10-24.
        image_token_start_index: Position where visual tokens begin in the
            input sequence (model / tokeniser dependent).
        image_token_length: Number of visual tokens produced by the vision
            encoder (e.g. 576 for LLaVA-1.5 with CLIP-ViT-L/14@336).
        gamma_percentile: Percentile of clean-calibration distance scores
            used as the detection threshold (gamma).  Paper default: 99.
        prune_threshold: Per-head attention threshold (tau).  Visual tokens
            whose max-head attention exceeds this value are pruned.
        neg_inf: Large negative value added to attention logits of pruned
            tokens.
    """

    detection_layers: List[int] = field(
        default_factory=lambda: list(range(10, 25))
    )
    image_token_start_index: int = 35
    image_token_length: int = 576
    gamma_percentile: float = 99.0
    prune_threshold: float = 0.02
    neg_inf: float = -1e38

    # --- populated after calibration ---
    dist_mu: Optional[List[float]] = None
    dist_std: Optional[List[float]] = None
    dist_thr: Optional[float] = None

    # --- runtime state (not serialized) ---

    @classmethod
    def from_yaml(cls, path: str) -> "CleanSightConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        data = {
            k: v
            for k, v in self.__dict__.items()
            if v is not None
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @property
    def is_calibrated(self) -> bool:
        return self.dist_mu is not None and self.dist_std is not None and self.dist_thr is not None

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the detection feature vector (num_layers * num_heads)."""
        # Actual dim depends on the model's num_heads; this is a helper
        # that returns the number of detection layers (heads are determined at runtime).
        return len(self.detection_layers)
