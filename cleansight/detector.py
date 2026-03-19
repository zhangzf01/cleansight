"""Anomaly detector: measures vision-to-text attention ratio and flags poisoned inputs."""

from typing import Dict, List, Optional

import numpy as np
import torch


class AttentionAnomalyDetector:
    """Detects backdoor-poisoned inputs via whitened L2 distance of
    per-head vision-to-text attention ratios.

    During **calibration** (``collect`` + ``fit``), clean-sample attention
    ratios are gathered to compute the reference mean / std and threshold.

    During **inference** (``score`` + ``is_poisoned``), new inputs are scored
    against the calibrated distribution.
    """

    def __init__(
        self,
        detection_layers: List[int],
        gamma_percentile: float = 99.0,
    ):
        self.detection_layers = detection_layers
        self.gamma_percentile = gamma_percentile

        # Calibration storage
        self._cal_features: List[np.ndarray] = []

        # Fitted parameters
        self.mu: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_attention_ratio(
        attn_weights: torch.Tensor,
        img_start: int,
        img_end: int,
    ) -> torch.Tensor:
        """Compute per-head vision-to-text attention ratio S^{l,h}.

        Args:
            attn_weights: Attention weights of shape ``[B, H, Q, K]``.
                Only the last query token (``[:, :, -1, :]``) is used.
            img_start: Start index of visual tokens in the key dimension.
            img_end: End index (exclusive) of visual tokens.

        Returns:
            Tensor of shape ``[H]`` with the ratio for each head.
        """
        attn_last = attn_weights[0, :, -1, :]  # [H, K]
        vision_attn = attn_last[:, img_start:img_end].sum(dim=-1)  # [H]
        prompt_attn = attn_last[:, img_end:].sum(dim=-1)           # [H]
        ratio = vision_attn / (prompt_attn + 1e-12)                # [H]
        return ratio

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def collect(self, feature: np.ndarray) -> None:
        """Store a single calibration feature vector (from a clean sample)."""
        self._cal_features.append(feature)

    def fit(self) -> None:
        """Fit the reference distribution from collected calibration features
        and compute the detection threshold at ``gamma_percentile``.
        """
        if len(self._cal_features) == 0:
            raise RuntimeError("No calibration features collected. Call collect() first.")
        feats = np.stack(self._cal_features, axis=0)  # [N, D]
        self.mu = feats.mean(axis=0)
        self.std = feats.std(axis=0)
        scores = np.array([self._whitened_l2(f) for f in feats])
        self.threshold = float(np.percentile(scores, self.gamma_percentile))

    # ------------------------------------------------------------------
    # Inference scoring
    # ------------------------------------------------------------------

    def score(self, feature: np.ndarray) -> float:
        """Compute the whitened L2 anomaly score for a feature vector."""
        if self.mu is None or self.std is None:
            raise RuntimeError("Detector is not calibrated. Call fit() first.")
        return self._whitened_l2(feature)

    def is_poisoned(self, feature: np.ndarray) -> bool:
        """Return ``True`` if the anomaly score exceeds the threshold."""
        if self.threshold is None:
            raise RuntimeError("Detector is not calibrated. Call fit() first.")
        return bool(self.score(feature) > self.threshold)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _whitened_l2(self, x: np.ndarray) -> float:
        z = (x - self.mu) / (self.std + 1e-8)
        return float(np.linalg.norm(z))

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "mu": self.mu.tolist() if self.mu is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "threshold": self.threshold,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.mu = np.array(state["mu"]) if state["mu"] is not None else None
        self.std = np.array(state["std"]) if state["std"] is not None else None
        self.threshold = state["threshold"]
