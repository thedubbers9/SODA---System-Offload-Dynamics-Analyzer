"""Minimal logical R/M/P/E/D buffers from grouped GEMM metadata (structural, not SSA).

Sizes are approximate (default fp16/bf16 element size) for architectural residency
estimation, not exact tensor recovery.

This is a minimal MoE-local reconstruction pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from soda.moe.moe_dataflow.ordering import StreamNode
from soda.moe.moe_dataflow.pairing import GemmPairRecord, PairingResult
from soda.moe.op_profile import _compute_hbm_fields, _dtype_bytes


def _dtype_bytes_for_precision(precision: str) -> int:
    return _dtype_bytes(precision)


def _activation_bytes_node(aten_op_name: str, input_dims: Any, precision: str) -> float:
    b = _dtype_bytes_for_precision(precision)
    h = _compute_hbm_fields(aten_op_name, input_dims or [], b)
    return float(h.get("activation_bytes", 0.0) or 0.0)


def _pair_shape(layer_ops: List[StreamNode], expand_ei: int, down_ei: int) -> Dict[str, Any]:
    ex = layer_ops[expand_ei]
    dn = layer_ops[down_ei]
    t0, h0, r0 = ex.grouped_mm_T, ex.grouped_mm_H, ex.grouped_mm_R
    t1, r1 = dn.grouped_mm_T, dn.grouped_mm_R
    t_match = t0 is not None and t1 is not None and t0 == t1
    return {
        "T_expand": t0,
        "H": h0,
        "R_expand": r0,
        "T_down": t1,
        "R_down": r1,
        "T_consistent": t_match,
        "bytes_formula_notes": (
            "P: T*H*elem; E: T*R0*elem (first grouped_mm output); "
            "D: T*R1*elem (second grouped_mm output); elem from precision unless trace dtype known."
        ),
    }


def _pair_buffer_bytes(
    layer_ops: List[StreamNode],
    pair: GemmPairRecord,
    precision: str,
) -> Dict[str, float]:
    elem = float(_dtype_bytes_for_precision(precision))
    ex = layer_ops[pair.expand_ei]
    dn = layer_ops[pair.down_ei]
    t = ex.grouped_mm_T or dn.grouped_mm_T
    h = ex.grouped_mm_H
    r0 = ex.grouped_mm_R
    r1 = dn.grouped_mm_R
    p_b = float(t * h * elem) if t and h else 0.0
    e_b = float(t * r0 * elem) if t and r0 else 0.0
    d_b = float(t * r1 * elem) if t and r1 else 0.0
    return {"P": p_b, "E": e_b, "D": d_b, "uncertain": 0.0 if (t and h and r0 and r1) else 1.0}


@dataclass
class ChainBuffers:
    layer_id: int
    anchor_ei: int
    buffers: List[Dict[str, Any]]
    shape_debug: List[Dict[str, Any]]


def build_chain_buffers(
    layer_ops: List[StreamNode],
    anchor_ei: int,
    pairing: PairingResult,
    *,
    precision: str,
) -> ChainBuffers:
    """Structural buffers: R, M, P_i, E_i, D_i (simulator-friendly dicts)."""
    buffers: List[Dict[str, Any]] = []
    shape_rows: List[Dict[str, Any]] = []

    gate = layer_ops[anchor_ei]
    r_bytes = _activation_bytes_node(gate.aten_op_name, gate.input_dims, precision)
    buffers.append(
        {
            "name": "R",
            "class": "routing_logits",
            "producer_ei": anchor_ei,
            "size_bytes_estimate": r_bytes,
            "size_formula": "op_profile activation_bytes heuristic on gate aten+input_dims",
        }
    )

    select_eis = [c.execution_index for c in pairing.classified if c.coarse_class == "routing_select"]
    meta_eis = [c.execution_index for c in pairing.classified if c.coarse_class == "routing_metadata"]
    m_ei = meta_eis[-1] if meta_eis else (select_eis[-1] if select_eis else anchor_ei)
    m_node = layer_ops[m_ei]
    m_bytes = _activation_bytes_node(m_node.aten_op_name, m_node.input_dims, precision)
    buffers.append(
        {
            "name": "M",
            "class": "routing_metadata",
            "producer_ei": m_ei,
            "size_bytes_estimate": m_bytes,
            "size_formula": "last metadata op if any else last select else gate (structural placeholder)",
        }
    )

    for pair in pairing.pairs:
        i = pair.pair_id
        ex_ei, dn_ei = pair.expand_ei, pair.down_ei
        pb = _pair_buffer_bytes(layer_ops, pair, precision)
        shape_info = _pair_shape(layer_ops, ex_ei, dn_ei)
        shape_rows.append(
            {
                "pair_id": i,
                "expand_ei": ex_ei,
                "down_ei": dn_ei,
                **shape_info,
                "estimated_bytes": {"P": pb["P"], "E": pb["E"], "D": pb["D"]},
                "uncertain_sizing": bool(pb.get("uncertain", 0)),
            }
        )
        buffers.append(
            {
                "name": f"P{i}",
                "class": "packed_input",
                "consumer_ei": ex_ei,
                "size_bytes_estimate": pb["P"],
                "size_formula": "T*H*elem from first grouped_mm of pair (see shape_debug)",
            }
        )
        buffers.append(
            {
                "name": f"E{i}",
                "class": "expert_intermediate",
                "producer_ei": ex_ei,
                "consumer_ei": dn_ei,
                "size_bytes_estimate": pb["E"],
                "size_formula": "T*R_expand*elem",
            }
        )
        buffers.append(
            {
                "name": f"D{i}",
                "class": "expert_output",
                "producer_ei": dn_ei,
                "size_bytes_estimate": pb["D"],
                "size_formula": "T*R_down*elem",
            }
        )

    return ChainBuffers(layer_id=gate.layer_id, anchor_ei=anchor_ei, buffers=buffers, shape_debug=shape_rows)


def all_chains_to_json(
    chains: List[Dict[str, Any]],
    *,
    ordering_source: str,
    order_note: str,
) -> Dict[str, Any]:
    return {
        "schema": "moe_minimal_chains_v1",
        "ordering_source": ordering_source,
        "ordering_note": order_note,
        "chains": chains,
    }


def all_buffers_to_json(buffer_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"schema": "moe_minimal_buffers_v1", "layers": buffer_chains}
