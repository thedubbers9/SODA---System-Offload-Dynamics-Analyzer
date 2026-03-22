# Abstraction: **anchor-centered local windows** over the ordered kernel stream.
#
# We anchor on ``_grouped_mm`` and other strong MoE signals, then grow a bounded
# neighborhood in execution order.  This reconstructs an **architecture-facing
# local MoE subgraph**, not a full-layer PDG: scratchpad opportunities live next
# to grouped expert GEMMs, not across arbitrary depth.
#
# **Experimental:** grouping is deliberately local; global full-program heuristics
# are avoided in favor of anchors that hardware can trust.
"""Discover :class:`PipelineGroup` windows around grouped expert and MoE anchors.

**Why anchors beat global labeling:** routing and packing kernels are only
interpretable **relative to** the grouped expert GEMM chain.  We therefore find
``aten::_grouped_mm`` (and gate / shared-expert GEMM anchors) first, then classify
the surrounding neighborhood — not the reverse.

Grouped expert kernels are the **natural center of the scratchpad opportunity**:
they couple expert-major weights with routed activations in one place.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from soda.moe.experimental.experimental_nodes import GROUPED_MM_ATEN
from soda.moe.experimental.experimental_types import KernelNode, PipelineGroup


DEFAULT_HALF_WINDOW = 28


def is_hard_terminator(n: KernelNode) -> bool:
    """Stop expanding a window at attention / norm boundaries."""
    if n.expert_type == "attention" or n.structural_role == "attention":
        return True
    if n.op_name in ("attn_bmm_kv", "attn_proj", "rmsnorm", "layernorm", "softmax"):
        return True
    return False


def is_strong_anchor(n: KernelNode) -> bool:
    """Kernels that can seed a local MoE residency window.

    We anchor on **grouped expert GEMMs** first — they are the scratchpad centerpiece.
    Gate and shared-expert GEMMs are strong secondary anchors.  When Inductor does
    not emit ``_grouped_mm``, dimension-tagged ``routed_expert_*`` structural roles
    still seed a window.  We deliberately do **not** treat every ``routed_expert``
    template as an anchor (that would merge unrelated templates into giant groups).
    """
    if n.aten_op_name == GROUPED_MM_ATEN:
        return True
    if n.op_name == "moe_gate_proj" or n.structural_role == "moe_gate":
        return True
    if n.structural_role in ("shared_expert_expand", "shared_expert_down"):
        return True
    if n.structural_role in ("routed_expert_expand", "routed_expert_down"):
        return True
    return False


def _expand_indices(
    layer_nodes: Sequence[KernelNode],
    anchor_pos: int,
    half_window: int,
) -> Tuple[int, int]:
    """Return ``[left, right]`` inclusive indices in ``layer_nodes`` around anchor."""
    n = len(layer_nodes)
    left = anchor_pos
    steps = 0
    while left > 0 and steps < half_window:
        if is_hard_terminator(layer_nodes[left - 1]):
            break
        left -= 1
        steps += 1

    right = anchor_pos
    steps = 0
    while right < n - 1 and steps < half_window:
        if is_hard_terminator(layer_nodes[right + 1]):
            break
        right += 1
        steps += 1
    return left, right


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [intervals[0]]
    for a, b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def discover_pipeline_groups(
    nodes: Sequence[KernelNode],
    half_window: int = DEFAULT_HALF_WINDOW,
) -> List[PipelineGroup]:
    """Build pipeline groups per layer from anchor-centered windows.

    Overlapping windows on the same layer merge into a single group so the
    simulator sees one connected MoE island rather than fragmented duplicates.
    """
    by_layer: Dict[int, List[KernelNode]] = {}
    for n in nodes:
        by_layer.setdefault(n.layer_id, []).append(n)

    groups: List[PipelineGroup] = []
    gid = 0

    for layer_id in sorted(by_layer):
        layer_nodes = sorted(by_layer[layer_id], key=lambda x: x.execution_index)
        raw_intervals: List[Tuple[int, int]] = []

        for i, kn in enumerate(layer_nodes):
            if not is_strong_anchor(kn):
                continue
            lo, hi = _expand_indices(layer_nodes, i, half_window)
            raw_intervals.append((lo, hi))

        if not raw_intervals:
            continue

        for lo, hi in _merge_intervals(raw_intervals):
            window = layer_nodes[lo : hi + 1]
            if not window:
                continue
            node_ids = [nn.node_id for nn in window]
            gkind = "moe_local_subgraph"
            if any(nn.aten_op_name == GROUPED_MM_ATEN for nn in window):
                gkind = "grouped_expert_moe_chain"
            conf, gnotes = _group_confidence(window)
            groups.append(
                PipelineGroup(
                    group_id=gid,
                    layer_id=layer_id,
                    start_execution_index=window[0].execution_index,
                    end_execution_index=window[-1].execution_index,
                    node_ids=node_ids,
                    group_kind=gkind,
                    confidence=conf,
                    notes=gnotes,
                )
            )
            gid += 1

    return groups


def assign_nodes_to_groups(
    nodes: List[KernelNode],
    groups: Sequence[PipelineGroup],
) -> None:
    """Set ``KernelNode.pipeline_group_id`` (last group wins if windows overlap)."""
    for g in groups:
        id_set = set(g.node_ids)
        for n in nodes:
            if n.node_id in id_set:
                n.pipeline_group_id = g.group_id


def _group_confidence(window: Sequence[KernelNode]) -> Tuple[str, str]:
    """``(confidence, notes)`` from local anchor composition."""
    has_g = any(n.aten_op_name == GROUPED_MM_ATEN for n in window)
    has_gate = any(n.op_name == "moe_gate_proj" or n.structural_role == "moe_gate" for n in window)
    roles_hint = {n.expert_type for n in window}
    if has_g and has_gate:
        return (
            "high",
            "Grouped MM + gate in same window; strong MoE scratchpad anchor.",
        )
    if has_g:
        return "medium", "Grouped MM present; routing/pack chain may be partial."
    if has_gate and "routed_expert" in roles_hint:
        return "medium", "Gate + routed expert templates without explicit grouped_mm in window."
    return "low", "MoE-related anchor window without grouped_mm; name/shape heuristics only."
