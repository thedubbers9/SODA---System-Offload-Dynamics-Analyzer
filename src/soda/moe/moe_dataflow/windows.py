"""Tiny chain windows: gate through last grouped GEMM (+ optional tail).

This is a minimal MoE-local reconstruction pass; windows are intentionally narrow
and anchor-first (no large pre-gate groups).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from soda.moe.moe_dataflow.anchors import is_gate_candidate
from soda.moe.moe_dataflow.ordering import GROUPED_MM_ATEN, StreamNode

# Include up to this many ops after the last _grouped_mm in the chain window.
TRAILING_OPS_AFTER_LAST_GEMM = 2


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
    """From a validated gate anchor, extend forward to the last grouped GEMM (+ tail).

    The scan stops before the **next** ``moe_gate_proj`` so a later MoE block in the
    same layer is not merged into this chain.
    """
    layer_id = layer_ops[anchor_ei].layer_id if layer_ops else 0
    next_gate = len(layer_ops)
    for j in range(anchor_ei + 1, len(layer_ops)):
        if is_gate_candidate(layer_ops[j]):
            next_gate = j
            break
    last_mm = anchor_ei
    for i in range(anchor_ei, next_gate):
        if _is_grouped_mm(layer_ops[i]):
            last_mm = i
    end_ei = min(len(layer_ops) - 1, last_mm + TRAILING_OPS_AFTER_LAST_GEMM)
    indices = list(range(anchor_ei, end_ei + 1))
    return ChainWindow(
        layer_id=layer_id,
        anchor_ei=anchor_ei,
        start_ei=anchor_ei,
        end_ei=end_ei,
        node_indices=indices,
    )
