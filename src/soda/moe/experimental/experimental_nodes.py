# Abstraction: turn classified kernel-database entries into **ordered KernelNode**
# records with byte estimates suitable for residency sizing.
#
# Differs from op_profile._compute_hbm_fields: we treat ``aten::_grouped_mm`` as a
# first-class grouped expert GEMM with an **explicit expert axis** in weights
# ``[E, H, R]`` (or transposes).  Without this, the routed MoE body can disappear
# into ``other`` byte heuristics and vanish as an architectural anchor.
#
# **Experimental:** broad ``expert_type`` from detect.py is a **hint** only; grouped
# shape parsing can stand alone for anchor discovery.
"""Build :class:`KernelNode` instances from the kernel DB + optional trace timestamps.

Grouped GEMM (``aten::_grouped_mm``) is the natural **center of the MoE scratchpad
opportunity**: one kernel touches expert-major weights and routed activations.
We parse activation ``[T, H]`` and weights ``[E, H, R]`` (or equivalent) to
expose ``num_experts=E``, ``input_dim=H``, ``output_dim=R`` for downstream
grouping and simulators — even when cardinality-based ``expert_type`` mislabels
the entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.moe.experimental.experimental_order import trace_first_ts_per_kernel_name
from soda.moe.experimental.experimental_types import KernelNode
from soda.moe.op_profile import _compute_hbm_fields, _dtype_bytes, _infer_structural_op_name
from soda.moe.op_profile import _ops_per_layer as ops_per_layer


GROUPED_MM_ATEN = "aten::_grouped_mm"


def _normalize_shape(shape: Any) -> List[int]:
    from soda.moe.op_profile import _normalize_shape as _ns

    return _ns(shape)


def _product(shape: Any) -> int:
    from soda.moe.op_profile import _product as _p

    return _p(shape)


def parse_grouped_mm_shapes(input_dims: Any) -> Optional[Dict[str, int]]:
    """Parse ``_grouped_mm`` (activation, weights) into expert-major semantics.

    **Convention (Inductor-style MoE):** activation ``[T, H]`` (token-grouped
    routed batch × hidden), weights ``[E, H, R]`` — **E** experts, **H** input
    dim, **R** output dim per expert.  Some builds use ``[E, R, H]``; we accept
    3D weight tensors by picking the dimension that matches activation hidden H.

    **Architectural point:** the expert axis **E** is explicit in the weight
    tensor; this is exactly the signal hardware wants to anchor scratchpad reuse
    (expert-major staging) on.
    """
    if not input_dims or len(input_dims) < 2:
        return None
    act = _normalize_shape(input_dims[0])
    w = _normalize_shape(input_dims[1])
    if len(act) < 2 or len(w) != 3:
        return None
    # Routed batch dimension T, hidden H for typical [T, H] activations.
    t = int(act[0])
    h_act = int(act[1])
    a0, a1, a2 = int(w[0]), int(w[1]), int(w[2])
    if h_act == a1:
        e, h, r = a0, a1, a2
    elif h_act == a2:
        e, r, h = a0, a1, a2
    else:
        e, h, r = a0, a1, a2
    return {"T": t, "H": h, "R": r, "E": e}


def compute_hbm_fields_with_grouped_mm(
    aten_op_name: str,
    input_dims: Any,
    dtype_bytes: int,
) -> Dict[str, float]:
    """Byte / FLOP estimates including dedicated ``_grouped_mm`` handling.

    Falls back to ``op_profile._compute_hbm_fields`` for standard GEMM ops; if that
    returns zeros and the op is grouped MM, use grouped shape math.
    """
    base = _compute_hbm_fields(aten_op_name, input_dims or [], dtype_bytes)
    if aten_op_name != GROUPED_MM_ATEN:
        return base
    g = parse_grouped_mm_shapes(input_dims)
    if not g:
        return base
    t, h, r, e = g["T"], g["H"], g["R"], g["E"]
    flops = 2 * t * h * r
    weight_bytes = float(e * h * r * dtype_bytes)
    activation_bytes = float((t * h + t * r) * dtype_bytes)
    hbm = weight_bytes + activation_bytes
    return {
        "flops": float(flops),
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "hbm_bytes": hbm,
        "kv_bytes": 0.0,
    }


def infer_experimental_op_name(
    aten_op_name: str,
    structural_role: str,
    input_dims: Any,
) -> str:
    """Human-readable op label; never collapses ``_grouped_mm`` into generic ``linear``."""
    if aten_op_name == GROUPED_MM_ATEN:
        return "grouped_expert_gemm"
    return _infer_structural_op_name(aten_op_name, structural_role, input_dims)


def _cta_count(entry: Dict) -> int:
    grid = (entry.get("kernel") or {}).get("grid") or [1, 1, 1]
    cta = 1
    for dim in grid:
        cta *= int(dim) if dim else 1
    return cta


def build_kernel_nodes(
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    trace_path: Optional[Path] = None,
    ordered_entries: Optional[List[Dict]] = None,
    ordering_source: str = "",
    order_note: str = "",
) -> Tuple[List[KernelNode], str, str]:
    """Emit ordered :class:`KernelNode` for all ``layer_id`` in [0, num_layers).

    If ``ordered_entries`` is provided, it must already be in execution order;
    otherwise callers should run :func:`experimental_order.order_classified_entries`
    first and pass the result (this avoids double-sorting when the pipeline owns
    ordering metadata).
    """
    from soda.moe.experimental.experimental_order import order_classified_entries

    if ordered_entries is None:
        ordered_entries, ordering_source, order_note = order_classified_entries(
            classified_kernels, trace_path
        )

    dtype_b = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
    num_layers = max(1, int(num_layers))

    first_ts: Dict[str, float] = {}
    if trace_path and Path(trace_path).is_file():
        first_ts = trace_first_ts_per_kernel_name(Path(trace_path))

    nodes: List[KernelNode] = []

    for layer_id in range(num_layers):
        layer_ei = 0
        for entry in ordered_entries:
            expert_type = entry.get("expert_type", "other")
            structural_role = entry.get("structural_role", "other")
            aten_op = entry.get("aten_op", {})
            aten_op_name = aten_op.get("name", "")
            input_dims = aten_op.get("input_dims", [])
            entry_id = entry.get("id", "")
            stats = entry.get("statistics", {})
            freq = int(stats.get("frequency", 1))
            latency_us = float(stats.get("avg_duration_us", 0.0) or 0.0)
            kn = (entry.get("kernel") or {}).get("name", "") or ""
            ts_us = first_ts.get(kn) if first_ts else None

            hbm_fields = compute_hbm_fields_with_grouped_mm(aten_op_name, input_dims, dtype_b)
            if entry_id in ncu_results:
                ncu = ncu_results[entry_id]
                ncu_hbm = float(
                    (ncu.get("hbm_read_bytes") or 0) + (ncu.get("hbm_write_bytes") or 0)
                )
                if ncu_hbm > 0:
                    hbm_fields = dict(hbm_fields)
                    hbm_fields["hbm_bytes"] = ncu_hbm

            is_shared = expert_type == "shared_expert"
            gshape = parse_grouped_mm_shapes(input_dims) if aten_op_name == GROUPED_MM_ATEN else None
            notes_parts: List[str] = []
            if gshape:
                notes_parts.append(
                    f"grouped_mm E={gshape['E']} H={gshape['H']} R={gshape['R']} T={gshape['T']}"
                )
            if expert_type == "other" and aten_op_name == GROUPED_MM_ATEN:
                notes_parts.append(
                    "expert_type=other but _grouped_mm shapes present; experimental path "
                    "still treats as MoE anchor"
                )

            ops_count = ops_per_layer(freq, num_layers) if expert_type != "routed_expert" else 0

            def emit_one() -> None:
                nonlocal layer_ei, nodes
                _op_name = infer_experimental_op_name(
                    aten_op_name, structural_role, input_dims
                )
                nodes.append(
                    KernelNode(
                        node_id=f"L{layer_id}:N{layer_ei}",
                        execution_index=layer_ei,
                        trace_ts_us=ts_us,
                        layer_id=layer_id,
                        source_entry_id=str(entry_id) if entry_id else None,
                        kernel_name=kn,
                        aten_op_name=aten_op_name,
                        op_name=_op_name,
                        input_dims=input_dims,
                        expert_type=expert_type,
                        structural_role=structural_role,
                        activation_bytes=float(hbm_fields.get("activation_bytes", 0.0)),
                        weight_bytes=float(hbm_fields.get("weight_bytes", 0.0)),
                        hbm_bytes=float(hbm_fields.get("hbm_bytes", 0.0)),
                        latency_us=latency_us,
                        cta_count=_cta_count(entry),
                        is_shared_expert=is_shared,
                        notes="; ".join(notes_parts),
                        grouped_mm_num_experts=gshape["E"] if gshape else None,
                        grouped_mm_input_dim=gshape["H"] if gshape else None,
                        grouped_mm_output_dim=gshape["R"] if gshape else None,
                    )
                )
                layer_ei += 1

            if ops_count > 0:
                for _ in range(ops_count):
                    emit_one()
            elif expert_type == "routed_expert":
                emit_one()

    return nodes, ordering_source, order_note
