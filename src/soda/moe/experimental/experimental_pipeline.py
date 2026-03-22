# This orchestrator **intentionally bypasses** the flat op-profile interface:
# ``op_profile.json`` discards the local producer-consumer structure scratchpad
# simulators need.  The experimental path loads ``kernel_database.json`` (+ optional
# ``trace.json`` + CLI/model overrides) and emits ``moe_dataflow_graph.json`` only
# through this pipeline — **no op_profile.json is required** as an intermediate.
#
# **Experimental:** optimized for architecture-facing MoE dataflow reconstruction;
# not API-stable with legacy op_profile consumers.
"""End-to-end experimental MoE dataflow reconstruction (kernel DB → JSON artifacts).

Inputs:
  * Classified kernel list (from ``classify_kernel_entries``) **or** raw kernels
    (this entrypoint expects the caller to pass classified kernels for dimension hints).
  * Optional ``trace_path`` for ordering.
  * ``num_layers``, ``precision``, optional NCU overrides.

Outputs (under ``moe_profile/`` or caller-provided ``output_dir``)::
  * moe_pipeline_nodes.json
  * moe_pipeline_groups.json
  * moe_dataflow_graph.json  ← simulator handoff
  * moe_dataflow.debug.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from soda.moe.dataflow import resolve_trace_path

from soda.moe.experimental.experimental_buffers import (
    compute_workspace_estimates,
    reconstruct_logical_buffers,
)
from soda.moe.experimental.experimental_export import emit_experimental_moe_artifacts
from soda.moe.experimental.experimental_groups import assign_nodes_to_groups, discover_pipeline_groups
from soda.moe.experimental.experimental_nodes import build_kernel_nodes
from soda.moe.experimental.experimental_roles import build_all_stage_nodes
def run_experimental_moe_dataflow(
    *,
    classified_kernels: List[dict],
    output_dir: Path,
    kernel_db_path: Optional[Path] = None,
    trace_path: Optional[Path] = None,
    num_layers: int = 1,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    moe_debug_log_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Run the experimental pipeline and return output path mapping."""
    output_dir = Path(output_dir)
    tp = trace_path
    if tp is None and kernel_db_path is not None:
        tp = resolve_trace_path(Path(kernel_db_path))

    nodes, ordering_source, order_note = build_kernel_nodes(
        classified_kernels,
        num_layers=num_layers,
        precision=precision,
        ncu_results=ncu_results,
        trace_path=tp,
    )

    groups = discover_pipeline_groups(nodes)
    assign_nodes_to_groups(nodes, groups)
    stage_nodes = build_all_stage_nodes(groups, nodes)
    buffers = reconstruct_logical_buffers(groups, stage_nodes)
    workspace = compute_workspace_estimates(groups, buffers)

    trace_used = str(tp) if tp and Path(tp).is_file() else None
    return emit_experimental_moe_artifacts(
        output_dir=output_dir,
        nodes=nodes,
        groups=groups,
        stage_nodes=stage_nodes,
        buffers=buffers,
        workspace=workspace,
        ordering_source=ordering_source,
        order_note=order_note,
        trace_path_used=trace_used,
        num_layers=num_layers,
        precision=precision,
        moe_debug_log_path=moe_debug_log_path,
    )
