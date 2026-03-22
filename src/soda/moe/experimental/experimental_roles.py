# Abstraction: **context-aware stage labels** (R/M/P/E/D pipeline roles) inside each
# local MoE window.
#
# Generic kernels (`copy_`, `index`, elementwise) are meaningless globally; they
# only become ``pack_input`` vs ``unpack_reduce`` **relative to** grouped expert
# GEMM frontiers.  Stage-role assignment is therefore **local-window based**.
#
# **Experimental:** we intentionally infer from ATen name + grouped shapes + trace-local
# order without requiring broad ``expert_type`` to be correct (``_grouped_mm`` may
# have been invisible to legacy classifiers).
"""Assign ``stage_role`` per node inside each :class:`PipelineGroup`.

Broad ``expert_type`` (shared_expert / routed_expert / …) from cardinality heuristics
is **useful but insufficient**: Inductor may emit ``aten::_grouped_mm`` while the
DB row remains ``other``.  This module may still classify grouped expert kernels
from the ATen op name and **grouped weight shape** (expert axis E) plus position
inside the anchor window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from soda.moe.experimental.experimental_nodes import GROUPED_MM_ATEN
from soda.moe.experimental.experimental_types import KernelNode, PipelineGroup, StageNode


_ROUTING_SELECT_ATEN = frozenset({
    "aten::topk",
    "aten::sort",
    "aten::argsort",
})

_ROUTING_META_ATEN = frozenset({
    "aten::cumsum",
    "aten::histc",
    "aten::arange",
    "aten::fill_",
    "aten::zero_",
    "aten::scatter",
    "aten::scatter_add",
})

_ELEMENTWISE_ATEN = frozenset({
    "aten::mul",
    "aten::add",
    "aten::div",
    "aten::sub",
    "aten::sigmoid",
})

_ACTIVATION_ATEN = frozenset({
    "aten::silu",
    "aten::gelu",
    "aten::relu",
    "aten::tanh",
})

_PACKISH_ATEN = frozenset({
    "aten::index",
    "aten::index_select",
    "aten::gather",
    "aten::cat",
    "aten::stack",
    "aten::copy_",
    "aten::reshape",
    "aten::view",
    "aten::transpose",
    "aten::permute",
})


@dataclass
class _LocalRoleCtx:
    """Scan state **within one PipelineGroup** (reset per group)."""

    seen_routing_logits: bool = False
    seen_routing_select: bool = False
    seen_routing_meta: bool = False
    seen_expert_expand: bool = False
    seen_expert_down: bool = False
    in_routing_region: bool = False
    in_expert_body: bool = False


def _is_grouped_mm_body(n: KernelNode) -> bool:
    """True if this kernel is the grouped expert GEMM anchor (shape or ATen name)."""
    if n.aten_op_name == GROUPED_MM_ATEN:
        return True
    if "grouped_mm" in n.aten_op_name:
        return True
    return False


def infer_stage_role(
    n: KernelNode,
    ctx: _LocalRoleCtx,
) -> Tuple[str, str]:
    """Return ``(stage_role, stage_confidence)`` for one node inside a group."""
    aten = n.aten_op_name or ""
    sr = n.structural_role or ""
    et = n.expert_type or ""
    op = n.op_name or ""

    if et == "shared_expert" and sr == "shared_expert_expand":
        ctx.in_expert_body = True
        return "shared_expert_expand", "high"
    if et == "shared_expert" and sr == "shared_expert_down":
        ctx.seen_expert_down = True
        ctx.in_expert_body = False
        return "shared_expert_down", "high"

    if sr == "moe_gate" or op == "moe_gate_proj":
        ctx.seen_routing_logits = True
        ctx.in_routing_region = True
        return "routing_logits", "high"

    if aten in _ROUTING_SELECT_ATEN:
        ctx.seen_routing_select = True
        ctx.in_routing_region = True
        return "routing_select", "high"

    if aten in _ROUTING_META_ATEN:
        ctx.seen_routing_meta = True
        ctx.in_routing_region = True
        return "routing_metadata", "medium"

    # Grouped expert GEMM: **do not** require et == routed_expert.
    # We anchor scratchpad modeling on this kernel; expand vs down follows order.
    if _is_grouped_mm_body(n):
        if not ctx.seen_expert_expand:
            ctx.seen_expert_expand = True
            ctx.in_routing_region = False
            ctx.in_expert_body = True
            conf = "high" if n.grouped_mm_num_experts is not None else "medium"
            return "expert_expand", conf
        if not ctx.seen_expert_down:
            ctx.seen_expert_down = True
            ctx.in_expert_body = False
            return "expert_down", "high"
        return "expert_down", "medium"

    if et == "routed_expert" and sr == "routed_expert_expand":
        ctx.seen_expert_expand = True
        ctx.in_routing_region = False
        ctx.in_expert_body = True
        return "expert_expand", "high"

    if et == "routed_expert" and sr == "routed_expert_down":
        ctx.seen_expert_down = True
        ctx.in_expert_body = False
        return "expert_down", "high"

    if aten in _ACTIVATION_ATEN or op == "activation":
        if ctx.in_expert_body or (ctx.seen_expert_expand and not ctx.seen_expert_down):
            return "expert_nonlinearity", "medium"
        return "other", "low"

    if aten in _ELEMENTWISE_ATEN:
        if ctx.in_expert_body and ctx.seen_expert_expand and not ctx.seen_expert_down:
            return "expert_nonlinearity", "medium"
        return "other", "low"

    if aten in _PACKISH_ATEN or op == "elementwise":
        if not ctx.seen_expert_expand and (ctx.in_routing_region or ctx.seen_routing_logits):
            return "pack_input", "medium"
        if ctx.seen_expert_down:
            return "unpack_reduce", "medium"
        if ctx.seen_expert_expand and not ctx.seen_expert_down:
            return "expert_nonlinearity", "low"
        return "other", "low"

    if op in ("attn_bmm_kv", "attn_proj") or et == "attention" or sr == "attention":
        return "other", "low"
    if op in ("rmsnorm", "layernorm", "softmax"):
        return "other", "low"

    return "other", "low"


def build_stage_nodes_for_group(
    group: PipelineGroup,
    node_map: Dict[str, KernelNode],
) -> List[StageNode]:
    """Ordered :class:`StageNode` list for one group with linear pred/succ links."""
    window = [node_map[nid] for nid in group.node_ids if nid in node_map]
    window.sort(key=lambda x: x.execution_index)
    ctx = _LocalRoleCtx()
    stages: List[StageNode] = []
    for n in window:
        role, conf = infer_stage_role(n, ctx)
        kdict = {
            "node_id": n.node_id,
            "layer_id": n.layer_id,
            "execution_index": n.execution_index,
            "trace_ts_us": n.trace_ts_us,
            "source_entry_id": n.source_entry_id,
            "kernel_name": n.kernel_name,
            "aten_op_name": n.aten_op_name,
            "op_name": n.op_name,
            "expert_type": n.expert_type,
            "structural_role": n.structural_role,
            "activation_bytes": n.activation_bytes,
            "weight_bytes": n.weight_bytes,
            "hbm_bytes": n.hbm_bytes,
            "latency_us": n.latency_us,
            "cta_count": n.cta_count,
            "is_shared_expert": n.is_shared_expert,
            "grouped_mm_num_experts": n.grouped_mm_num_experts,
            "grouped_mm_input_dim": n.grouped_mm_input_dim,
            "grouped_mm_output_dim": n.grouped_mm_output_dim,
            "notes": n.notes,
        }
        stages.append(
            StageNode(
                node_id=n.node_id,
                group_id=group.group_id,
                stage_role=role,
                stage_confidence=conf,
                kernel=kdict,
            )
        )

    for i, s in enumerate(stages):
        if i > 0:
            s.predecessor_node_ids.append(stages[i - 1].node_id)
        if i + 1 < len(stages):
            s.successor_node_ids.append(stages[i + 1].node_id)

    return stages


def build_all_stage_nodes(
    groups: Sequence[PipelineGroup],
    nodes: Sequence[KernelNode],
) -> List[StageNode]:
    """Flattened stage nodes for all groups (execution order within each group)."""
    node_map = {n.node_id: n for n in nodes}
    out: List[StageNode] = []
    for g in groups:
        out.extend(build_stage_nodes_for_group(g, node_map))
    return out
