"""Tiny chain windows: gate through last grouped GEMM (+ optional tail).

This is a minimal MoE-local reconstruction pass; windows are intentionally narrow
and anchor-first (no large pre-gate groups).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from soda.moe.moe_dataflow.anchors import is_gate_candidate
from soda.moe.moe_dataflow.ordering import GROUPED_MM_ATEN, StreamNode

# Trailing mm/linear after the last _grouped_mm (at most one op, only if it matches).
_POST_EXPERT_ATEN = frozenset(
    {
        "aten::linear",
        "aten::mm",
        "aten::bmm",
        "aten::addmm",
        "aten::matmul",
        "aten::_scaled_mm",
    }
)


@dataclass
class ChainWindow:
    """Indices inclusive: ``start_ei`` … ``end_ei`` within one layer."""

    layer_id: int
    anchor_ei: int
    start_ei: int
    end_ei: int
    node_indices: List[int] = field(default_factory=list)


def _is_grouped_mm(n: StreamNode) -> bool:
    return n.aten_op_name == GROUPED_MM_ATEN


def build_chain_window(layer_ops: List[StreamNode], anchor_ei: int) -> ChainWindow:
    """Anchor-first window: gate through last ``_grouped_mm`` in this block (+ optional tail).

    Stops before the next ``moe_gate_proj``. Does not walk backward before the gate.
    Shared-expert ``_grouped_mm`` ops that appear **before** this anchor are never
    included.
    """
    return extract_minimal_routed_window(layer_ops, anchor_ei)


def extract_minimal_routed_window(layer_ops: List[StreamNode], anchor_idx: int) -> ChainWindow:
    """Forward-only window from a validated gate to the routed-expert grouped-GEMM tail."""
    layer_id = layer_ops[anchor_idx].layer_id if layer_ops else 0
    next_gate = len(layer_ops)
    for j in range(anchor_idx + 1, len(layer_ops)):
        if is_gate_candidate(layer_ops[j]):
            next_gate = j
            break
    last_mm = anchor_idx
    for i in range(anchor_idx, next_gate):
        if _is_grouped_mm(layer_ops[i]):
            last_mm = i
    end_ei = last_mm
    if last_mm + 1 < next_gate and last_mm + 1 < len(layer_ops):
        nxt = layer_ops[last_mm + 1]
        if (nxt.aten_op_name or "") in _POST_EXPERT_ATEN:
            end_ei = last_mm + 1
    indices = list(range(anchor_idx, end_ei + 1))
    return ChainWindow(
        layer_id=layer_id,
        anchor_ei=anchor_idx,
        start_ei=anchor_idx,
        end_ei=end_ei,
        node_indices=indices,
    )


def minimal_routed_window_stream_nodes(
    layer_ops: List[StreamNode],
    anchor_idx: int,
) -> List[StreamNode]:
    """Same window as :func:`extract_minimal_routed_window`, as concrete nodes."""
    cw = extract_minimal_routed_window(layer_ops, anchor_idx)
    return [layer_ops[i] for i in cw.node_indices]
