# Abstraction: **JSON + debug text** for human audit and simulator ingest.
#
# The canonical simulator handoff is ``moe_dataflow_graph.json``; the split
# ``moe_pipeline_nodes.json`` / ``moe_pipeline_groups.json`` files mirror common
# tooling patterns (nodes table vs group summaries).
#
# **Experimental:** this replaces the idea of stuffing everything into
# ``op_profile.json`` — that flat schema cannot carry local producer-consumer
# structure.
"""Write ``moe_dataflow_graph.json``, companion JSON files, and ``moe_dataflow.debug.txt``.

The simulator should prefer **moe_dataflow_graph.json** (metadata + ordered nodes +
groups with nested buffers and workspace).  The debug log makes architectural
interpretation **auditable**: anchors, boundaries, roles, buffer reconstruction,
and confidence rationale.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from soda.moe.detect import append_moe_op_profile_debug

from soda.moe.experimental.experimental_types import (
    KernelNode,
    LogicalBuffer,
    PipelineGroup,
    StageNode,
    WorkspaceEstimate,
)


def _kernel_node_to_dict(n: KernelNode) -> Dict[str, Any]:
    d = asdict(n)
    return d


def write_moe_pipeline_nodes_json(
    path: Path,
    nodes: Sequence[KernelNode],
    stage_by_node: Dict[str, Tuple[str, str, int]],
    metadata: Dict[str, Any],
) -> None:
    """``moe_pipeline_nodes.json`` — ordered kernel nodes with group/stage labels."""
    rows: List[Dict[str, Any]] = []
    for n in sorted(nodes, key=lambda x: (x.layer_id, x.execution_index)):
        sr, sc, _gid = stage_by_node.get(n.node_id, ("", "", -1))
        d = _kernel_node_to_dict(n)
        d["stage_role"] = sr or None
        d["stage_confidence"] = sc or None
        rows.append(d)
    path.write_text(
        json.dumps({"metadata": metadata, "nodes": rows}, indent=2),
        encoding="utf-8",
    )


def write_moe_pipeline_groups_json(
    path: Path,
    groups: Sequence[PipelineGroup],
    stage_by_group: Dict[int, List[StageNode]],
    metadata: Dict[str, Any],
) -> None:
    """``moe_pipeline_groups.json`` — groups with embedded stage-enriched nodes."""
    payload_groups: List[Dict[str, Any]] = []
    for g in groups:
        stages = stage_by_group.get(g.group_id, [])
        payload_groups.append(
            {
                **asdict(g),
                "nodes": [
                    {
                        "node_id": s.node_id,
                        "stage_role": s.stage_role,
                        "stage_confidence": s.stage_confidence,
                        "predecessor_node_ids": s.predecessor_node_ids,
                        "successor_node_ids": s.successor_node_ids,
                        "kernel": s.kernel,
                    }
                    for s in sorted(stages, key=lambda x: x.kernel.get("execution_index", 0))
                ],
            }
        )
    path.write_text(
        json.dumps({"metadata": metadata, "groups": payload_groups}, indent=2),
        encoding="utf-8",
    )


def write_moe_dataflow_graph_json(
    path: Path,
    metadata: Dict[str, Any],
    nodes: Sequence[KernelNode],
    groups: Sequence[PipelineGroup],
    stage_nodes: Sequence[StageNode],
    buffers: Sequence[LogicalBuffer],
    workspace: Sequence[WorkspaceEstimate],
    stage_by_group: Dict[int, List[StageNode]],
) -> None:
    """Single simulator handoff document."""
    ws_by_gid = {w.group_id: w for w in workspace}
    group_payload: List[Dict[str, Any]] = []
    for g in groups:
        w = ws_by_gid.get(g.group_id)
        stages = stage_by_group.get(g.group_id, [])
        buf_list = [asdict(b) for b in buffers if b.group_id == g.group_id]
        group_payload.append(
            {
                "group_id": g.group_id,
                "layer_id": g.layer_id,
                "start_execution_index": g.start_execution_index,
                "end_execution_index": g.end_execution_index,
                "group_kind": g.group_kind,
                "confidence": g.confidence,
                "notes": g.notes,
                "nodes": [
                    {
                        **s.kernel,
                        "stage_role": s.stage_role,
                        "stage_confidence": s.stage_confidence,
                        "predecessor_node_ids": s.predecessor_node_ids,
                        "successor_node_ids": s.successor_node_ids,
                    }
                    for s in sorted(stages, key=lambda x: x.kernel.get("execution_index", 0))
                ],
                "buffers": buf_list,
                "workspace_estimate": asdict(w) if w else None,
            }
        )

    ordered_nodes = [
        _kernel_node_to_dict(n)
        for n in sorted(nodes, key=lambda x: (x.layer_id, x.execution_index))
    ]

    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "ordered_nodes": ordered_nodes,
                "groups": group_payload,
                "buffers": [asdict(b) for b in buffers],
                "workspace_estimates": [asdict(w) for w in workspace],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_moe_dataflow_debug(
    path: Path,
    *,
    ordering_source: str,
    order_note: str,
    nodes: Sequence[KernelNode],
    groups: Sequence[PipelineGroup],
    stage_nodes: Sequence[StageNode],
    buffers: Sequence[LogicalBuffer],
    workspace: Sequence[WorkspaceEstimate],
) -> None:
    """Human-readable audit of reconstruction decisions (not machine-parseable)."""
    lines = [
        "# moe_dataflow.debug.txt — experimental MoE dataflow reconstruction audit",
        "# Purpose: make anchors, windows, stage roles, and buffer sizes reviewable.",
        "",
        f"ordering_source={ordering_source}",
        f"ordering_note={order_note}",
        "",
        "--- anchors (grouped_mm / gate / shared expert) ---",
    ]
    for n in nodes:
        if n.aten_op_name == "aten::_grouped_mm" or n.op_name == "moe_gate_proj":
            lines.append(
                f"  {n.node_id} aten={n.aten_op_name} op={n.op_name} "
                f"ei={n.execution_index} layer={n.layer_id} notes={n.notes}"
            )
        elif n.structural_role in ("shared_expert_expand", "shared_expert_down", "moe_gate"):
            lines.append(
                f"  {n.node_id} structural={n.structural_role} op={n.op_name} "
                f"ei={n.execution_index} layer={n.layer_id}"
            )

    lines.append("")
    lines.append("--- ordered nodes (layer, ei, stage, op, aten, entry) ---")
    sn_map = {s.node_id: s for s in stage_nodes}
    for n in sorted(nodes, key=lambda x: (x.layer_id, x.execution_index)):
        s = sn_map.get(n.node_id)
        role = s.stage_role if s else "-"
        lines.append(
            f"  L{n.layer_id} ei={n.execution_index} stage={role} op={n.op_name} "
            f"aten={n.aten_op_name} entry={n.source_entry_id} group={n.pipeline_group_id}"
        )

    lines.append("")
    lines.append("--- pipeline groups ---")
    for g in groups:
        lines.append(
            f"  group {g.group_id} layer={g.layer_id} ei=[{g.start_execution_index},"
            f"{g.end_execution_index}] kind={g.group_kind} conf={g.confidence} "
            f"n_nodes={len(g.node_ids)} notes={g.notes}"
        )

    lines.append("")
    lines.append("--- stage roles (within groups) ---")
    for s in sorted(stage_nodes, key=lambda x: (x.group_id, x.kernel.get("execution_index", 0))):
        lines.append(
            f"  g={s.group_id} {s.node_id} role={s.stage_role} conf={s.stage_confidence}"
        )

    lines.append("")
    lines.append("--- logical buffers ---")
    for b in buffers:
        lines.append(
            f"  {b.buffer_id} class={b.buffer_class} group={b.group_id} "
            f"prod={b.producer_node_id} cons={b.consumer_node_ids} "
            f"size={b.size_bytes_estimate:.0f} method={b.size_estimation_method} "
            f"conf={b.confidence}"
        )

    lines.append("")
    lines.append("--- workspace ---")
    for w in workspace:
        lines.append(
            f"  group {w.group_id}: staged_peak={w.peak_bytes_staged:.0f} "
            f"conservative_peak={w.peak_bytes_conservative:.0f} "
            f"R={w.R_bytes:.0f} M={w.M_bytes:.0f} P={w.P_bytes:.0f} "
            f"E={w.E_bytes:.0f} D={w.D_bytes:.0f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_experimental_moe_artifacts(
    *,
    output_dir: Path,
    nodes: List[KernelNode],
    groups: List[PipelineGroup],
    stage_nodes: List[StageNode],
    buffers: List[LogicalBuffer],
    workspace: List[WorkspaceEstimate],
    ordering_source: str,
    order_note: str,
    trace_path_used: Optional[str],
    num_layers: int,
    precision: str,
    moe_debug_log_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write all experimental JSON + debug; return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema": "experimental_moe_dataflow_v1",
        "ordering_source": ordering_source,
        "ordering_warning": order_note,
        "trace_path_used": trace_path_used,
        "num_layers": num_layers,
        "precision": precision,
        "simulator_handoff": (
            "Use moe_dataflow_graph.json for scratchpad/residency simulation; "
            "this path bypasses op_profile.json structure intentionally."
        ),
        "confidence_note": (
            "Confidence labels describe reconstruction fidelity of the local MoE graph, "
            "not model accuracy or performance."
        ),
    }

    stage_by_group: Dict[int, List[StageNode]] = {}
    for s in stage_nodes:
        stage_by_group.setdefault(s.group_id, []).append(s)

    stage_by_node: Dict[str, Tuple[str, str, int]] = {}
    for s in stage_nodes:
        stage_by_node[s.node_id] = (s.stage_role, s.stage_confidence, s.group_id)

    p_nodes = output_dir / "moe_pipeline_nodes.json"
    p_groups = output_dir / "moe_pipeline_groups.json"
    p_graph = output_dir / "moe_dataflow_graph.json"
    p_dbg = output_dir / "moe_dataflow.debug.txt"

    write_moe_pipeline_nodes_json(p_nodes, nodes, stage_by_node, metadata)
    write_moe_pipeline_groups_json(p_groups, groups, stage_by_group, metadata)
    write_moe_dataflow_graph_json(
        p_graph, metadata, nodes, groups, stage_nodes, buffers, workspace, stage_by_group
    )
    write_moe_dataflow_debug(
        p_dbg,
        ordering_source=ordering_source,
        order_note=order_note,
        nodes=nodes,
        groups=groups,
        stage_nodes=stage_nodes,
        buffers=buffers,
        workspace=workspace,
    )

    append_moe_op_profile_debug(
        moe_debug_log_path,
        f"[moe.experimental] wrote {p_graph.name}, {p_nodes.name}, {p_groups.name}, {p_dbg.name}",
    )

    return {
        "moe_dataflow_graph": p_graph,
        "moe_pipeline_nodes": p_nodes,
        "moe_pipeline_groups": p_groups,
        "moe_dataflow_debug": p_dbg,
    }
