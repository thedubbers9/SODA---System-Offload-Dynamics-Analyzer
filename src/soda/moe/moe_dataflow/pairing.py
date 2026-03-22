"""Grouped GEMM discovery, consecutive pairing, and coarse window classification.

Logical labels ``expert_pair_expand`` / ``expert_pair_down`` are minimal model roles
per pair (first / second grouped GEMM); they are not claimed to match exact MLP
semantics.

This is a minimal MoE-local reconstruction pass — no pack/unpack/shared-expert
modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from soda.moe.moe_dataflow.ordering import GROUPED_MM_ATEN, StreamNode

_SELECT_MARKERS = ("topk", "sort")
_METADATA_MARKERS = ("histc", "cumsum", "fill_")
_GEMM_FOR_POST = frozenset(
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
class ClassifiedOp:
    execution_index: int
    aten_op_name: str
    op_name: str
    kernel_entry_name: str
    coarse_class: str  # routing_logits | routing_select | routing_metadata | grouped_expert_gemm | post_expert_candidate | unclassified
    gemm_pair_index: Optional[int] = None
    gemm_logical_role: Optional[str] = None  # expert_pair_expand | expert_pair_down


@dataclass
class GemmPairRecord:
    pair_id: int
    expand_ei: int
    down_ei: int


@dataclass
class PairingResult:
    classified: List[ClassifiedOp]
    grouped_mm_eis: List[int]
    pairs: List[GemmPairRecord]
    odd_gemm_warning: bool


def _classify_window_ops(
    layer_ops: List[StreamNode],
    window_indices: List[int],
    anchor_ei: int,
    grouped_mm_eis: List[int],
) -> List[ClassifiedOp]:
    last_mm = grouped_mm_eis[-1] if grouped_mm_eis else anchor_ei
    out: List[ClassifiedOp] = []
    for idx in window_indices:
        n = layer_ops[idx]
        a = (n.aten_op_name or "").lower()
        coarse = "unclassified"
        pair_idx: Optional[int] = None
        role: Optional[str] = None

        if idx == anchor_ei:
            coarse = "routing_logits"
        elif any(m in a for m in _SELECT_MARKERS):
            coarse = "routing_select"
        elif any(m in a for m in _METADATA_MARKERS):
            coarse = "routing_metadata"
        elif n.aten_op_name == GROUPED_MM_ATEN:
            coarse = "grouped_expert_gemm"
            if grouped_mm_eis:
                try:
                    pos = grouped_mm_eis.index(idx)
                except ValueError:
                    pos = -1
                if pos >= 0:
                    pair_idx = pos // 2
                    role = "expert_pair_expand" if pos % 2 == 0 else "expert_pair_down"
        elif idx > last_mm and n.aten_op_name in _GEMM_FOR_POST:
            coarse = "post_expert_candidate"

        out.append(
            ClassifiedOp(
                execution_index=idx,
                aten_op_name=n.aten_op_name,
                op_name=n.op_name,
                kernel_entry_name=n.kernel_entry_name,
                coarse_class=coarse,
                gemm_pair_index=pair_idx,
                gemm_logical_role=role,
            )
        )
    return out


def pair_grouped_gemms(grouped_mm_eis: List[int]) -> tuple[List[GemmPairRecord], bool]:
    """Pair ``(0,1), (2,3), ...``; drop trailing odd GEMM with warning flag."""
    pairs: List[GemmPairRecord] = []
    odd = False
    g = list(grouped_mm_eis)
    if len(g) % 2 == 1:
        odd = True
        g = g[:-1]
    for i in range(0, len(g), 2):
        pairs.append(GemmPairRecord(pair_id=i // 2, expand_ei=g[i], down_ei=g[i + 1]))
    return pairs, odd


def analyze_chain_window(
    layer_ops: List[StreamNode],
    window_indices: List[int],
    anchor_ei: int,
) -> PairingResult:
    grouped_mm_eis = [i for i in window_indices if layer_ops[i].aten_op_name == GROUPED_MM_ATEN]
    pairs, odd = pair_grouped_gemms(grouped_mm_eis)
    classified = _classify_window_ops(layer_ops, window_indices, anchor_ei, grouped_mm_eis)
    return PairingResult(
        classified=classified,
        grouped_mm_eis=grouped_mm_eis,
        pairs=pairs,
        odd_gemm_warning=odd,
    )


def pairing_result_to_json_dict(pr: PairingResult) -> Dict[str, Any]:
    return {
        "grouped_mm_execution_indices": list(pr.grouped_mm_eis),
        "pairs": [
            {"pair_id": p.pair_id, "expert_pair_expand_ei": p.expand_ei, "expert_pair_down_ei": p.down_ei}
            for p in pr.pairs
        ],
        "odd_unpaired_grouped_mm_dropped": pr.odd_gemm_warning,
        "ops": [
            {
                "execution_index": c.execution_index,
                "aten_op_name": c.aten_op_name,
                "op_name": c.op_name,
                "kernel_entry_name": c.kernel_entry_name,
                "coarse_class": c.coarse_class,
                "gemm_pair_index": c.gemm_pair_index,
                "gemm_logical_role": c.gemm_logical_role,
            }
            for c in pr.classified
        ],
    }
