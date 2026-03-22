"""Deterministic per-layer execution ordering for kernel-database templates.

Ordering is primarily first GPU kernel timestamp per cleaned kernel name from
``trace.json``, with fallback to kernel DB list order (weak proxy — noted in debug).

This is a minimal MoE-local reconstruction pass; async multi-stream schedules are
not solved beyond this best-effort ordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.common.data import clean_kernel_name
from soda.moe.op_profile import _infer_structural_op_name, _ops_per_layer


GROUPED_MM_ATEN = "aten::_grouped_mm"


def trace_first_ts_per_kernel_name(trace_path: Path) -> Dict[str, float]:
    """Map cleaned GPU kernel name → first Chrome-trace ``ts`` (microseconds)."""
    try:
        data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    events = data.get("traceEvents") or []
    first: Dict[str, float] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        name = clean_kernel_name(ev.get("name", "") or "")
        if not name:
            continue
        ts = float(ev.get("ts", 0.0))
        if name not in first or ts < first[name]:
            first[name] = ts
    return first


def order_classified_entries(
    classified_kernels: List[dict],
    trace_path: Optional[Path],
) -> Tuple[List[dict], str, str]:
    """Return ``(ordered_entries, ordering_source, note_or_warning)``."""
    if not classified_kernels:
        return [], "empty", ""

    if trace_path is not None and Path(trace_path).is_file():
        first_ts = trace_first_ts_per_kernel_name(Path(trace_path))

        def sort_key(e: dict) -> Tuple[float, int, str]:
            kn = (e.get("kernel") or {}).get("name", "") or ""
            ts = first_ts.get(kn, float("inf"))
            rank = int(e.get("rank", 10**9))
            eid = str(e.get("id", ""))
            return (ts, rank, eid)

        ordered = sorted(classified_kernels, key=sort_key)
        note = (
            "Ordering uses first GPU kernel timestamp per cleaned kernel name from "
            "trace.json; async streams may still reorder real execution."
        )
        return ordered, "trace_first_kernel_ts", note

    ordered = list(classified_kernels)
    warn = (
        "WARNING: No trace.json — using kernel_database.json list order "
        "(often sorted by aggregate duration, not program order)."
    )
    return ordered, "kernel_db_list_fallback", warn


def _normalize_shape(shape: Any) -> List[int]:
    from soda.moe.op_profile import _normalize_shape as _ns

    return _ns(shape)


def parse_grouped_mm_shapes(input_dims: Any) -> Optional[Dict[str, int]]:
    """Parse ``_grouped_mm`` into T, H, R, E (Inductor-style MoE convention)."""
    if not input_dims or len(input_dims) < 2:
        return None
    act = _normalize_shape(input_dims[0])
    w = _normalize_shape(input_dims[1])
    if len(act) < 2 or len(w) != 3:
        return None
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


@dataclass
class StreamNode:
    """One slot in the per-layer deterministic op stream."""

    layer_id: int
    execution_index: int
    source_entry_id: Optional[str]
    kernel_entry_name: str
    aten_op_name: str
    op_name: str
    structural_role: str
    expert_type: str
    input_dims: Any
    grouped_mm_T: Optional[int] = None
    grouped_mm_H: Optional[int] = None
    grouped_mm_R: Optional[int] = None
    grouped_mm_E: Optional[int] = None


def build_ordered_stream_per_layer(
    ordered_entries: List[Dict[str, Any]],
    *,
    num_layers: int,
) -> Tuple[List[List[StreamNode]], str]:
    """One ordered list of :class:`StreamNode` per layer (same expansion rules as op_profile path)."""
    num_layers = max(1, int(num_layers))
    streams: List[List[StreamNode]] = [[] for _ in range(num_layers)]

    for layer_id in range(num_layers):
        layer_ei = 0
        for entry in ordered_entries:
            expert_type = entry.get("expert_type", "other")
            structural_role = entry.get("structural_role", "other")
            aten_op = entry.get("aten_op", {})
            aten_op_name = aten_op.get("name", "")
            input_dims = aten_op.get("input_dims", [])
            entry_id = entry.get("id", "")
            kn = (entry.get("kernel") or {}).get("name", "") or ""
            freq = int((entry.get("statistics") or {}).get("frequency", 1))

            op_name = _infer_structural_op_name(aten_op_name, structural_role, input_dims)
            gshape = parse_grouped_mm_shapes(input_dims) if aten_op_name == GROUPED_MM_ATEN else None

            def emit_one() -> None:
                nonlocal layer_ei
                streams[layer_id].append(
                    StreamNode(
                        layer_id=layer_id,
                        execution_index=layer_ei,
                        source_entry_id=str(entry_id) if entry_id else None,
                        kernel_entry_name=kn,
                        aten_op_name=aten_op_name,
                        op_name=op_name,
                        structural_role=structural_role,
                        expert_type=expert_type,
                        input_dims=input_dims,
                        grouped_mm_T=gshape["T"] if gshape else None,
                        grouped_mm_H=gshape["H"] if gshape else None,
                        grouped_mm_R=gshape["R"] if gshape else None,
                        grouped_mm_E=gshape["E"] if gshape else None,
                    )
                )
                layer_ei += 1

            ops_count = _ops_per_layer(freq, num_layers) if expert_type != "routed_expert" else 0
            if ops_count > 0:
                for _ in range(ops_count):
                    emit_one()
            elif expert_type == "routed_expert":
                emit_one()

    return streams, "expanded_per_layer_template_stream"
