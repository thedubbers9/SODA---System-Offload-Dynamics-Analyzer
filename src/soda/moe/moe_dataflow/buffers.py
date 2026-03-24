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


def _pair_shape(layer_ops: List[StreamNode], gemm0_ei: int, gemm1_ei: int) -> Dict[str, Any]:
    ex = layer_ops[gemm0_ei]
    dn = layer_ops[gemm1_ei]
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
            "P_i: T*H*elem from first grouped_mm inputs; "
            "E_i: T*R0*elem (gemm0 output); D_i: T*R1*elem (gemm1 output); "
            "elem from precision unless trace dtype known."
        ),
    }


def _pair_buffer_bytes(
    layer_ops: List[StreamNode],
    pair: GemmPairRecord,
    precision: str,
) -> Dict[str, float]:
    elem = float(_dtype_bytes_for_precision(precision))
    ex = layer_ops[pair.gemm0_ei]
    dn = layer_ops[pair.gemm1_ei]
    t = ex.grouped_mm_T or dn.grouped_mm_T
    h = ex.grouped_mm_H
    r0 = ex.grouped_mm_R
    r1 = dn.grouped_mm_R
    p_b = float(t * h * elem) if t and h else 0.0
    e_b = float(t * r0 * elem) if t and r0 else 0.0
    d_b = float(t * r1 * elem) if t and r1 else 0.0
    p_el = float(t * h) if t and h else 0.0
    e_el = float(t * r0) if t and r0 else 0.0
    d_el = float(t * r1) if t and r1 else 0.0
    return {
        "P": p_b,
        "E": e_b,
        "D": d_b,
        "P_elements": p_el,
        "E_elements": e_el,
        "D_elements": d_el,
        "uncertain": 0.0 if (t and h and r0 and r1) else 1.0,
    }


@dataclass
class ChainBuffers:
    layer_id: int
    anchor_ei: int
    buffers: List[Dict[str, Any]]
    shape_debug: List[Dict[str, Any]]


def build_minimal_buffers(
    layer_ops: List[StreamNode],
    anchor_ei: int,
    pairing: PairingResult,
    *,
    precision: str,
) -> ChainBuffers:
    """Logical R, M, P_i, E_i, D_i from structural metadata only (no SSA recovery)."""
    return build_chain_buffers(layer_ops, anchor_ei, pairing, precision=precision)


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
            "elements_estimate": None,
            "bytes_estimate": r_bytes,
            "size_bytes_estimate": r_bytes,
            "size_formula": "activation_bytes heuristic on gate aten+input_dims (approximate)",
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
            "class": "routing_select_metadata",
            "producer_ei": m_ei,
            "elements_estimate": None,
            "bytes_estimate": m_bytes,
            "size_bytes_estimate": m_bytes,
            "size_formula": "last routing_metadata ei if any else last routing_select else gate (structural placeholder)",
        }
    )

    for pair in pairing.pairs:
        i = pair.pair_id
        g0_ei, g1_ei = pair.gemm0_ei, pair.gemm1_ei
        pb = _pair_buffer_bytes(layer_ops, pair, precision)
        shape_info = _pair_shape(layer_ops, g0_ei, g1_ei)
        shape_rows.append(
            {
                "pair_id": i,
                "gemm0_ei": g0_ei,
                "gemm1_ei": g1_ei,
                **shape_info,
                "estimated_bytes": {"P": pb["P"], "E": pb["E"], "D": pb["D"]},
                "elements_estimate": {"P": pb["P_elements"], "E": pb["E_elements"], "D": pb["D_elements"]},
                "uncertain_sizing": bool(pb.get("uncertain", 0)),
            }
        )
        buffers.append(
            {
                "name": f"P{i}",
                "class": "pair_gemm0_input",
                "consumer_ei": g0_ei,
                "elements_estimate": pb["P_elements"],
                "bytes_estimate": pb["P"],
                "size_bytes_estimate": pb["P"],
                "size_formula": "T*H*elem from grouped GEMM0 shape metadata (approximate)",
            }
        )
        buffers.append(
            {
                "name": f"E{i}",
                "class": "expert_intermediate",
                "producer_ei": g0_ei,
                "consumer_ei": g1_ei,
                "elements_estimate": pb["E_elements"],
                "bytes_estimate": pb["E"],
                "size_bytes_estimate": pb["E"],
                "size_formula": "T*R0*elem from grouped GEMM0 output dims (approximate)",
            }
        )
        buffers.append(
            {
                "name": f"D{i}",
                "class": "expert_output",
                "producer_ei": g1_ei,
                "elements_estimate": pb["D_elements"],
                "bytes_estimate": pb["D"],
                "size_bytes_estimate": pb["D"],
                "size_formula": "T*R1*elem from grouped GEMM1 output dims (approximate)",
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
        "schema": "moe_minimal_chains_v2",
        "ordering_source": ordering_source,
        "ordering_note": order_note,
        "chains": chains,
    }


def all_buffers_to_json(buffer_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"schema": "moe_minimal_buffers_v2", "layers": buffer_chains}
