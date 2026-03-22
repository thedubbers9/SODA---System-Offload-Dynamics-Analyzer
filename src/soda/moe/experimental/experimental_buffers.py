# Abstraction: **logical buffers** R/M/P/E/D and **workspace scalars** per group.
#
# These buffers do not appear as explicit tensors in the trace; they are reconstructed
# so the simulator can reason about **residency of intermediate conceptual tensors**
# rather than isolated per-kernel byte lines.
#
# **Experimental:** producer/consumer edges are explicit for lifetime modeling; sizes
# use max activation bytes among stage-matched nodes (documented per buffer).
"""Reconstruct :class:`LogicalBuffer` graphs and :class:`WorkspaceEstimate` per group.

**Simulator semantics:** each ``LogicalBuffer`` ties a **buffer_class** (R/M/P/E/D) to
one producer kernel and zero or more consumer kernels, plus an execution-index
interval.  The simulator cares about **buffer lifetimes across kernels**, not only
kernel identities.

**Workspace formulas (architecture-facing approximations, not exact allocators):**
  * ``peak_bytes_staged`` = R + M + max(P, E, D)  — P/E/D assumed staged.
  * ``peak_bytes_conservative`` = R + M + max(P+E, E+D, P+D) — adjacent buffers may overlap.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from soda.moe.experimental.experimental_types import (
    LogicalBuffer,
    PipelineGroup,
    StageNode,
    WorkspaceEstimate,
)


def _max_act_from_stages(stages: Iterable[StageNode]) -> float:
    return max((float(s.kernel.get("activation_bytes", 0) or 0) for s in stages), default=0.0)


def _first_stage(stages: List[StageNode]) -> Optional[StageNode]:
    if not stages:
        return None
    return min(stages, key=lambda s: int(s.kernel.get("execution_index", 0)))


def reconstruct_logical_buffers(
    groups: Sequence[PipelineGroup],
    stage_nodes: Sequence[StageNode],
) -> List[LogicalBuffer]:
    """Build R/M/P/E/D buffers with explicit producer/consumer node ids per group."""
    by_group: Dict[int, List[StageNode]] = {}
    for s in stage_nodes:
        by_group.setdefault(s.group_id, []).append(s)

    buffers: List[LogicalBuffer] = []
    bid = 0

    for g in groups:
        gn = by_group.get(g.group_id, [])
        if not gn:
            continue

        def with_roles(roles: frozenset) -> List[StageNode]:
            return [s for s in gn if s.stage_role in roles]

        r_st = with_roles(frozenset({"routing_logits", "routing_select"}))
        m_st = with_roles(frozenset({"routing_metadata"}))
        p_st = with_roles(frozenset({"pack_input"}))
        e_st = with_roles(
            frozenset({"expert_expand", "expert_nonlinearity", "shared_expert_expand"})
        )
        d_st = with_roles(frozenset({"expert_down", "shared_expert_down"}))
        u_st = with_roles(frozenset({"unpack_reduce"}))

        conf = g.confidence

        def make_buf(
            cls: str,
            producers: List[StageNode],
            consumers: List[StageNode],
            resident: bool,
            buf_conf: str,
            note: str,
            method: str,
        ) -> Optional[LogicalBuffer]:
            nonlocal bid
            if not producers:
                return None
            prod = _first_stage(producers)
            if prod is None:
                return None
            cons_list = [x for x in consumers if x is not None]
            all_n = producers + cons_list
            start = min(int(x.kernel.get("execution_index", 0)) for x in all_n)
            end = max(int(x.kernel.get("execution_index", 0)) for x in all_n)
            size = _max_act_from_stages(producers)
            if size <= 0 and cons_list:
                size = _max_act_from_stages(cons_list)
            buf = LogicalBuffer(
                buffer_id=f"G{g.group_id}:B{bid}",
                layer_id=g.layer_id,
                group_id=g.group_id,
                buffer_class=cls,
                producer_node_id=prod.node_id,
                consumer_node_ids=[x.node_id for x in cons_list],
                start_execution_index=start,
                end_execution_index=end,
                size_bytes_estimate=float(size),
                size_estimation_method=method,
                resident_candidate=resident,
                confidence=buf_conf,
                notes=note,
            )
            bid += 1
            return buf

        # R: routing logits/select; consumed when building metadata, packed inputs, unpack.
        b_r = make_buf(
            "R",
            r_st,
            m_st + p_st + u_st,
            True,
            conf,
            "Logical routing tensor (logits/selection); not a single profiler tensor.",
            "max_activation_bytes_among_routing_logits_select",
        )
        if b_r:
            buffers.append(b_r)

        # M: small metadata — we take **max** among metadata-stage nodes (conservative
        # if one kernel over-reports); summing would double-count shared scratch.
        b_m = make_buf(
            "M",
            m_st,
            p_st + u_st,
            True,
            conf,
            "Routing metadata (indices, counts, offsets); max-bytes rule avoids double count.",
            "max_activation_bytes_among_routing_metadata",
        )
        if b_m:
            buffers.append(b_m)

        b_p = make_buf(
            "P",
            p_st,
            e_st,
            True,
            conf,
            "Packed expert-major activations feeding expert GEMMs (incl. grouped_mm).",
            "max_activation_bytes_among_pack_input",
        )
        if b_p:
            buffers.append(b_p)

        b_e = make_buf(
            "E",
            e_st,
            d_st,
            True,
            conf,
            "Expert MLP intermediate; nonlinearities assumed to update E in-place.",
            "max_activation_bytes_among_expand_and_nonlinearity",
        )
        if b_e:
            buffers.append(b_e)

        b_d = make_buf(
            "D",
            d_st,
            u_st,
            True,
            conf,
            "Expert outputs before scatter/reduce back to token layout.",
            "max_activation_bytes_among_expert_down",
        )
        if b_d:
            buffers.append(b_d)

    return buffers


def compute_workspace_estimates(
    groups: Sequence[PipelineGroup],
    buffers: Sequence[LogicalBuffer],
) -> List[WorkspaceEstimate]:
    """One :class:`WorkspaceEstimate` per group from reconstructed buffers."""
    out: List[WorkspaceEstimate] = []
    for g in groups:
        mine = {b.buffer_class: b for b in buffers if b.group_id == g.group_id}

        def sz(cls: str) -> float:
            b = mine.get(cls)
            return float(b.size_bytes_estimate) if b else 0.0

        r_v, m_v = sz("R"), sz("M")
        p_v, e_v, d_v = sz("P"), sz("E"), sz("D")
        staged = r_v + m_v + max(p_v, e_v, d_v)
        conservative = r_v + m_v + max(p_v + e_v, e_v + d_v, p_v + d_v)
        out.append(
            WorkspaceEstimate(
                group_id=g.group_id,
                layer_id=g.layer_id,
                R_bytes=r_v,
                M_bytes=m_v,
                P_bytes=p_v,
                E_bytes=e_v,
                D_bytes=d_v,
                peak_bytes_staged=staged,
                peak_bytes_conservative=conservative,
                notes=(
                    "staged: R+M+max(P,E,D); conservative: R+M+max(P+E,E+D,P+D). "
                    "Not cycle-accurate dynamic lifetimes."
                ),
            )
        )
    return out
