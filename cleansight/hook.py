"""Monkey-patching utilities to inject CleanSight into any HuggingFace attention module.

Instead of complex hook registration, we replace the ``forward()`` method
of each attention module with a thin wrapper that:

1. Computes raw attention logits (pre-softmax).
2. On detection layers during prefill, extracts the vision-to-text ratio.
3. After the last detection layer, runs anomaly detection.
4. If poisoned, builds a pruning mask and applies it (current + all later layers).

This keeps the implementation model-agnostic while remaining simple and debuggable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .defense import CleanSightDefense


def _find_decoder_attn_modules(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Walk common HuggingFace model structures to find attention sub-modules."""
    for attr_path in [
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
        "transformer.h",
    ]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue

        results = []
        for idx, layer in enumerate(obj):
            for name in ["self_attn", "attn", "attention"]:
                attn = getattr(layer, name, None)
                if attn is not None:
                    results.append((idx, attn))
                    break
        if results:
            return results

    raise ValueError(
        "Could not locate decoder attention modules automatically. "
        "Supply them via the `attn_modules` argument."
    )


def patch_attention_forward(
    attn_module: nn.Module,
    layer_idx: int,
    defense: "CleanSightDefense",
) -> None:
    """Replace ``attn_module.forward`` with a CleanSight-aware version.

    The original forward is saved as ``attn_module._original_forward``.
    """
    attn_module._original_forward = attn_module.forward
    attn_module._cs_layer_idx = layer_idx
    attn_module._cs_defense = defense

    def cleansight_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Any]:
        mod = attn_module
        d = mod._cs_defense
        idx = mod._cs_layer_idx
        cfg = d.config

        bsz, q_len, _ = hidden_states.size()
        is_prefill = q_len > 1
        need_detection = is_prefill and idx in cfg.detection_layers
        need_pruning = d.pruner.is_active

        # If this layer needs neither detection nor pruning, use original forward
        if not need_detection and not need_pruning:
            return mod._original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        # --- Eager attention with CleanSight intervention ---
        num_heads = mod.num_heads
        head_dim = mod.head_dim
        num_kv_heads = mod.num_key_value_heads
        num_kv_groups = num_heads // num_kv_heads

        query_states = mod.q_proj(hidden_states)
        key_states = mod.k_proj(hidden_states)
        value_states = mod.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # RoPE
        cos, sin = mod.rotary_emb(value_states, position_ids)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache
        past_kv = getattr(mod, "past_key_value", past_key_value)
        if past_kv is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_kv.update(
                key_states, value_states, mod.layer_idx, cache_kwargs
            )

        # GQA expansion
        from transformers.models.llama.modeling_llama import repeat_kv
        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)

        # Raw attention logits
        attn_logits = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            attn_logits = attn_logits + causal_mask

        # --- CleanSight logic ---
        img_start = cfg.image_token_start_index
        img_end = img_start + cfg.image_token_length

        # Step 1: Detection — compute attention weights and extract ratios
        if need_detection:
            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

            ratio = d.detector.compute_attention_ratio(attn_weights, img_start, img_end)
            d._layer_ratios[idx] = ratio.detach().cpu()

            # On the last detection layer, run full detection
            if idx == cfg.detection_layers[-1]:
                d._run_detection(attn_weights, attn_logits, img_start, img_end)

        # Step 2: Pruning — apply mask to attention logits
        if d.pruner.is_active:
            attn_logits = d.pruner.apply(attn_logits)

        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=mod.attention_dropout, training=mod.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
        attn_output = mod.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_kv

    attn_module.forward = cleansight_forward


def unpatch_attention_forward(attn_module: nn.Module) -> None:
    """Restore the original forward method."""
    if hasattr(attn_module, "_original_forward"):
        attn_module.forward = attn_module._original_forward
        del attn_module._original_forward
