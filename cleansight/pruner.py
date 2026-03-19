"""Attention pruner: suppresses suspicious visual tokens by masking attention logits."""

from typing import Optional, Tuple

import torch


class AttentionPruner:
    """Identifies and suppresses trigger-bearing visual tokens.

    When a poisoned input is detected, visual tokens whose per-head
    attention weight exceeds ``prune_threshold`` are collected into a
    *union mask* across all heads.  In all subsequent layers, a large
    negative bias is added to the attention logits at those positions,
    effectively zeroing their softmax contribution.
    """

    def __init__(
        self,
        prune_threshold: float = 0.02,
        neg_inf: float = -1e38,
    ):
        self.prune_threshold = prune_threshold
        self.neg_inf = neg_inf

        # Runtime state: boolean mask [H, K] indicating tokens to suppress
        self._token_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Building the mask
    # ------------------------------------------------------------------

    def build_mask(
        self,
        attn_weights: torch.Tensor,
        img_start: int,
        img_end: int,
    ) -> None:
        """Build the global token-pruning mask from the detection layer.

        Args:
            attn_weights: Attention weights ``[B, H, Q, K]`` at the
                last detection layer (after softmax).
            img_start: Start index of visual tokens.
            img_end: End index (exclusive) of visual tokens.
        """
        attn_last = attn_weights[0, :, -1, :]  # [H, K]
        H, K = attn_last.shape

        vision_attn = attn_last[:, img_start:img_end]  # [H, N_vis]
        head_token_mask = vision_attn > self.prune_threshold  # [H, N_vis]

        # Expand to full key dimension
        full_mask = torch.zeros(H, K, dtype=torch.bool, device=attn_last.device)
        full_mask[:, img_start:img_end] = head_token_mask

        # Union across heads: if any head flags a token, suppress it for all
        token_union = full_mask.any(dim=0)  # [K]
        self._token_mask = token_union.unsqueeze(0).expand(H, K).detach().cpu()

    # ------------------------------------------------------------------
    # Applying the mask
    # ------------------------------------------------------------------

    def apply(
        self,
        attn_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the pruning mask to raw attention logits (pre-softmax).

        Args:
            attn_logits: Raw attention scores ``[B, H, Q, K_cur]``.

        Returns:
            Modified attention logits with pruned positions set to ``neg_inf``.
        """
        if self._token_mask is None:
            return attn_logits

        B, H, Q, K_cur = attn_logits.shape
        device = attn_logits.device

        mask = self._token_mask.to(device)
        H_mask, K_mask = mask.shape

        if H_mask != H:
            raise ValueError(f"Head mismatch: mask has {H_mask} heads, got {H}")

        # Pad if the current sequence is longer than the cached mask
        if K_mask < K_cur:
            pad = torch.zeros(H, K_cur - K_mask, dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        else:
            mask = mask[:, :K_cur]

        # Create additive mask
        bias = torch.zeros_like(attn_logits)
        bool_mask = mask.unsqueeze(0).unsqueeze(2).expand(B, H, Q, K_cur)
        bias[bool_mask] = self.neg_inf
        return attn_logits + bias

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._token_mask is not None

    def reset(self) -> None:
        """Clear the pruning mask (call between samples)."""
        self._token_mask = None
