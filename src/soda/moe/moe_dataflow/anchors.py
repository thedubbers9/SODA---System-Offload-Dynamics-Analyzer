"""Strict routed-expert gate anchor validation (tiny forward window only).

This is a minimal MoE-local reconstruction pass: it does not reconstruct the full
graph. Only ``moe_gate_proj``-like ops that co-occur with topk/sort and
``_grouped_mm`` within a short forward window are accepted as anchors.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from soda.moe.moe_dataflow.ordering import GROUPED_MM_ATEN, StreamNode

# Explicit small forward window (requirements: between 10 and 16 ops).
ANCHOR_VALIDATION_FORWARD_OPS = 12

_SELECT_SUBSTR = ("topk", "sort")
_METADATA_SUBSTR = ("histc", "cumsum", "fill_")


def _aten(n: StreamNode) -> str:
    return n.aten_op_name or ""


def is_gate_candidate(n: StreamNode) -> bool:
    """True only for routed-expert gate projection (not every gate-like op)."""
    return n.op_name == "moe_gate_proj"


def _is_select_op(n: StreamNode) -> bool:
    a = _aten(n).lower()
    return any(s in a for s in _SELECT_SUBSTR)


def _is_grouped_mm(n: StreamNode) -> bool:
    return n.aten_op_name == GROUPED_MM_ATEN


def _is_attention_barrier(n: StreamNode) -> bool:
    """Heuristic: attention kernels between gate and grouped GEMM invalidate the anchor."""
    if n.structural_role == "attention":
        return True
    if n.op_name in ("attn_proj", "attn_bmm_kv"):
        return True
    a = _aten(n)
    if "attention" in a.lower() or a in ("aten::scaled_dot_product_attention",):
        return True
    return False


def validate_gate_anchor(
    layer_ops: List[StreamNode],
    anchor_ei: int,
) -> Tuple[bool, str]:
    """Return ``(accepted, reason)`` for a candidate gate at ``anchor_ei``."""
    if anchor_ei < 0 or anchor_ei >= len(layer_ops):
        return False, "invalid execution index"
    gate = layer_ops[anchor_ei]
    if not is_gate_candidate(gate):
        return False, "not moe_gate_proj"

    end = min(len(layer_ops), anchor_ei + ANCHOR_VALIDATION_FORWARD_OPS)
    window = layer_ops[anchor_ei:end]

    has_select = any(_is_select_op(n) for n in window)
    if not has_select:
        return False, "no nearby topk/sort"

    first_mm_idx: Optional[int] = None
    for i, n in enumerate(window):
        if _is_grouped_mm(n):
            first_mm_idx = anchor_ei + i
            break
    if first_mm_idx is None:
        return False, "no nearby _grouped_mm"

    for j in range(anchor_ei + 1, first_mm_idx):
        if _is_attention_barrier(layer_ops[j]):
            return False, "unrelated attention block between gate and grouped GEMM"

    return True, "ok"


def find_gate_candidates(layer_ops: List[StreamNode]) -> List[int]:
    """Execution indices of all ``moe_gate_proj`` nodes in the layer."""
    return [n.execution_index for n in layer_ops if is_gate_candidate(n)]
