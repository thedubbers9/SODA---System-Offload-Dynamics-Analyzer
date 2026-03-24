"""Grouped GEMM pairing and coarse roles inside the minimal routed-expert window only.

Per-pair slots ``pair_N_gemm0`` / ``pair_N_gemm1`` (semantic aliases: pair_expand /
pair_down **within that pair only**). No global "first grouped GEMM in layer =
expand" rule.
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
    coarse_class: str  # routing_logits | routing_select | routing_metadata | grouped_expert_gemm | post_expert_candidate | unknown_within_window
    gemm_pair_index: Optional[int] = None
    pair_gemm_slot: Optional[str] = None  # gemm0 | gemm1 within the pair
    pair_role_label: Optional[str] = None  # e.g. pair_0_gemm0


@dataclass
class GemmPairRecord:
    pair_id: int
    gemm0_ei: int
    gemm1_ei: int


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
        coarse = "unknown_within_window"
        pair_idx: Optional[int] = None
        slot: Optional[str] = None
        pr_label: Optional[str] = None

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
                    slot = "gemm0" if pos % 2 == 0 else "gemm1"
                    pr_label = f"pair_{pair_idx}_gemm0" if pos % 2 == 0 else f"pair_{pair_idx}_gemm1"
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
                pair_gemm_slot=slot,
                pair_role_label=pr_label,
            )
        )
    return out


def pair_grouped_gemms(grouped_mm_eis: List[int]) -> tuple[List[GemmPairRecord], bool]:
    """Pair consecutive grouped GEMMs: (0,1), (2,3), …; drop trailing odd with warning."""
    pairs: List[GemmPairRecord] = []
    odd = False
    g = list(grouped_mm_eis)
    if len(g) % 2 == 1:
        odd = True
        g = g[:-1]
    for i in range(0, len(g), 2):
        pairs.append(GemmPairRecord(pair_id=i // 2, gemm0_ei=g[i], gemm1_ei=g[i + 1]))
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
    """Verbose pairing dump (optional / archival). Minimal JSON uses flat chain records."""
    return {
        "grouped_mm_execution_indices": list(pr.grouped_mm_eis),
        "gemm_pairs": [
            {
                "pair_id": p.pair_id,
                "gemm0_ei": p.gemm0_ei,
                "gemm1_ei": p.gemm1_ei,
                "pair_expand_ei": p.gemm0_ei,
                "pair_down_ei": p.gemm1_ei,
            }
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
                "pair_gemm_slot": c.pair_gemm_slot,
                "pair_role_label": c.pair_role_label,
            }
            for c in pr.classified
        ],
    }
