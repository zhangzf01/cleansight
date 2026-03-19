"""Main CleanSight defense: a training-free, plug-and-play test-time defense
against backdoor attacks on Large Vision-Language Models.

Usage
-----
::

    from cleansight import CleanSightConfig, CleanSightDefense

    cfg = CleanSightConfig(
        detection_layers=list(range(10, 25)),
        image_token_start_index=35,
        image_token_length=576,
    )
    defense = CleanSightDefense(cfg)

    # 1. Patch the model
    defense.patch(model)

    # 2. Calibrate on clean samples
    defense.set_mode("calibrate")
    for batch in clean_loader:
        model.generate(**batch)
    defense.fit()
    defense.save("cleansight_stats.yaml")

    # 3. Defend at test time
    defense.set_mode("defend")
    for batch in test_loader:
        defense.reset()          # clear per-sample state
        output = model.generate(**batch)
        print(f"Poisoned: {defense.was_poisoned}")
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import CleanSightConfig
from .detector import AttentionAnomalyDetector
from .pruner import AttentionPruner


class CleanSightDefense:
    """Orchestrates detection and pruning for test-time backdoor defense.

    Parameters
    ----------
    config : CleanSightConfig
        Defense hyper-parameters.
    """

    def __init__(self, config: CleanSightConfig):
        self.config = config
        self.detector = AttentionAnomalyDetector(
            detection_layers=config.detection_layers,
            gamma_percentile=config.gamma_percentile,
        )
        self.pruner = AttentionPruner(
            prune_threshold=config.prune_threshold,
            neg_inf=config.neg_inf,
        )

        # Load pre-calibrated stats if available
        if config.is_calibrated:
            self.detector.load_state_dict({
                "mu": config.dist_mu,
                "std": config.dist_std,
                "threshold": config.dist_thr,
            })

        self.mode: str = "off"  # "off", "calibrate", "defend"
        self._layer_ratios: Dict[int, torch.Tensor] = {}
        self._was_poisoned: bool = False
        self._patched_modules: List[nn.Module] = []

    # ------------------------------------------------------------------
    # Model patching
    # ------------------------------------------------------------------

    def patch(
        self,
        model: nn.Module,
        attn_modules: Optional[List[Tuple[int, nn.Module]]] = None,
    ) -> None:
        """Patch a model's attention layers with CleanSight-aware forward.

        Parameters
        ----------
        model : nn.Module
            The LVLM (e.g. ``LlavaForConditionalGeneration``).
        attn_modules : list of (layer_idx, attention_module), optional
            If not provided, auto-detected from the model structure.
        """
        from .hook import _find_decoder_attn_modules, patch_attention_forward

        if attn_modules is None:
            attn_modules = _find_decoder_attn_modules(model)

        for layer_idx, attn_mod in attn_modules:
            patch_attention_forward(attn_mod, layer_idx, self)
            self._patched_modules.append(attn_mod)

        # Force eager attention so we can intercept logits
        if hasattr(model, "config"):
            model.config._attn_implementation = "eager"
        # Also check nested configs (e.g. LLaVA wraps the LLM config)
        for attr in ["text_config", "language_config"]:
            nested = getattr(model.config, attr, None)
            if nested is not None:
                nested._attn_implementation = "eager"

    def unpatch(self) -> None:
        """Restore all patched attention modules to their original forward."""
        from .hook import unpatch_attention_forward
        for mod in self._patched_modules:
            unpatch_attention_forward(mod)
        self._patched_modules.clear()

    # ------------------------------------------------------------------
    # Mode control
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set the operating mode: ``"off"``, ``"calibrate"``, or ``"defend"``."""
        if mode not in ("off", "calibrate", "defend"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode

    # ------------------------------------------------------------------
    # Per-sample state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-sample state. Call before processing each new input."""
        self._layer_ratios.clear()
        self._was_poisoned = False
        self.pruner.reset()

    @property
    def was_poisoned(self) -> bool:
        """Whether the last processed sample was flagged as poisoned."""
        return self._was_poisoned

    # ------------------------------------------------------------------
    # Detection (called internally by the hook)
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        attn_weights: torch.Tensor,
        attn_logits: torch.Tensor,
        img_start: int,
        img_end: int,
    ) -> None:
        """Run anomaly detection after the last detection layer.

        Called internally by the patched attention forward.
        """
        feat_list = [
            self._layer_ratios[l]
            for l in self.config.detection_layers
            if l in self._layer_ratios
        ]
        if not feat_list:
            return

        feature = torch.cat(feat_list, dim=0).numpy()

        if self.mode == "calibrate":
            self.detector.collect(feature)
        elif self.mode == "defend" and self.detector.threshold is not None:
            score = self.detector.score(feature)
            if self.detector.is_poisoned(feature):
                self._was_poisoned = True
                self.pruner.build_mask(attn_weights, img_start, img_end)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Fit the detector on collected calibration features.

        Call after running all clean calibration samples through the model
        in ``"calibrate"`` mode.
        """
        self.detector.fit()
        # Sync back to config
        self.config.dist_mu = self.detector.mu.tolist()
        self.config.dist_std = self.detector.std.tolist()
        self.config.dist_thr = self.detector.threshold

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save calibrated config (including mu/std/threshold) to YAML."""
        self.config.to_yaml(path)

    @classmethod
    def load(cls, path: str) -> "CleanSightDefense":
        """Load a pre-calibrated defense from a YAML config file."""
        config = CleanSightConfig.from_yaml(path)
        return cls(config)

    def save_stats(self, path: str) -> None:
        """Save detector statistics to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.detector.state_dict(), f, indent=2)

    def load_stats(self, path: str) -> None:
        """Load detector statistics from a JSON file."""
        with open(path) as f:
            state = json.load(f)
        self.detector.load_state_dict(state)
        self.config.dist_mu = state["mu"]
        self.config.dist_std = state["std"]
        self.config.dist_thr = state["threshold"]
