# Copyright and architecture note:
#
# **Pipeline status:** The default ``--moe-profile`` path does **not** call this
# module. Use ``--moe-legacy-dataflow`` (or ``emit_dataflow_artifacts=True`` in
# ``generate_op_profile``) to emit ``dataflow_profile.json`` / broad
# ``PipelineGroup`` heuristics. For routed-expert chains, use
# ``soda.moe.moe_dataflow`` (minimal anchor-first parser).
#
# This module introduces an **architecture-facing approximation** of local MoE
# dataflow between classified kernel-database entries.  It is intentionally *not*
# an exact tensor-level dependence graph recovered from PyTorch: SODA's
# kernel_database.json aggregates unique (op, kernel) identities and loses true
# SSA-style value flow.  What we reconstruct instead is a **small logical buffer
# graph** (R/M/P/E/D) spanning short, contiguous MoE micro-pipelines so a
# downstream **architectural simulator** can reason about:
#
#   • producer → consumer adjacency between stages
#   • approximate live ranges of transient MoE tensors
#   • whether staged vs. conservative overlap models of scratchpad / controlled-L2
#     residency are plausible for chains like:
#       pack → grouped_mm → activation → grouped_mm → unpack
#
# **Directly known from SODA today**
#   • expert_type / structural_role from detect.py (dimension + cardinality heuristics)
#   • ATen op names and input_dims on each aggregated kernel entry
#   • Per-entry frequency, grid, and optional NCU byte overrides (via op_profile path)
#
# **Reconstructed heuristically**
#   • Execution order when trace.json is absent (kernel DB is sorted by total
#     duration, not program order — see ordering metadata we emit).
#   • dataflow_role for generic ops (copy_, index, elementwise) using **local pipeline
#     context** (before/after expert GEMMs).
#   • Contiguous PipelineGroups (gap-tolerant) over MoE-ish anchors.
#   • LogicalBuffer identities, sizes (max activation bytes among role-matched nodes),
#     and producer/consumer node IDs.
#
# **What remains approximate**
#   • Actual GPU schedule, stream interleaving, and aliasing of buffers.
#   • Exact footprint of routing metadata vs. logits (we bound via reported bytes).
#   • Whether two kernels truly touch the same physical allocation.
#
# Real scratchpad opportunities depend on **execution adjacency**.  When true trace
# order is unavailable, we reconstruct a plausible local order from classified
# kernel DB entries (optionally ordered by first GPU kernel timestamp in trace.json),
# but that is a **weaker approximation** — we label this explicitly in exported JSON.
#
"""MoE logical dataflow: pipeline nodes, groups, and reconstructed buffers.

This module sits **between** detect.py (classification) and op_profile.py (flat
per-op summaries).  pipeline.py orchestrates when artifacts are written.

Outputs (under ``moe_profile/``)::

    op_pipeline.json       — ordered PipelineNode list (execution + dataflow_role)
    dataflow_profile.json  — simulator handoff: nodes, groups, buffers, workspace
    dataflow.debug.txt     — human-readable reconstruction audit trail

``op_profile.json`` remains the **aggregated** per-layer operator summary sorted by
``op_name`` for stable diffs; it is intentionally insufficient for scratchpad modeling
because it discards execution order and producer/consumer structure.  The simulator
should prefer ``dataflow_profile.json`` (and ``op_pipeline.json``) for residency work.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from soda.common.data import clean_kernel_name

from soda.moe.detect import append_moe_op_profile_debug
from soda.moe.op_profile import (
    _compute_hbm_fields,
    _infer_structural_op_name,
    _ops_per_layer,
    _dtype_bytes,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineNode:
    """One **ordered** occurrence of a kernel / op template in a reconstructed MoE pipeline.

    Unlike a flat ``op_profile.json`` row (which may aggregate many invocations of the
    same ``op_name`` and is sorted for stability), a PipelineNode preserves **where**
    this template sits in the per-layer execution sequence.  That ordering is the
    substrate on which we infer ``dataflow_role`` and later attach logical buffers.

    **Abstraction level:** one node ≈ one expanded logical invocation slot matching
    ``generate_op_profile`` expansion rules (layer-local frequency expansion plus
    per-layer routed-expert templates).  It is still **not** one GPU thread block.

    **Why separate from op_profile:** scratchpad reuse is a property of **buffer
    lifetimes between kernels**, not of isolated operator byte totals.
    """

    node_id: str
    layer_id: int
    execution_index: int
    source_entry_id: Optional[str]
    op_name: str
    aten_op_name: str
    expert_type: str
    structural_role: str
    dataflow_role: str
    activation_bytes: float
    weight_bytes: float
    hbm_bytes: float
    cta_count: int
    latency_us: float
    is_shared_expert: bool
    profile_layer_id: int
    pipeline_group_id: Optional[int] = None
    ordering_note: str = ""


@dataclass
class LogicalBuffer:
    """A **reconstructed** logical intermediate tensor for MoE scratchpad modeling.

    Buffers R/M/P/E/D do **not** appear as explicit rows in the kernel database; they
    are architecture-facing abstractions inferred from stage roles (routing, pack,
    expert MLP, unpack).  Multiple kernels may read/write the same logical buffer.

    **Abstraction level:** sufficient to reason about **live intervals** ``[start,
    end]`` in execution-index space within a layer group — the same level a hardware
    simulator needs when deciding whether P, E, or D can stay resident in a 3D SRAM
    or controlled-L2 partition across a short chain.

    **Why not use op_profile alone:** op_profile loses **which producer fed which
    consumer**; this record makes adjacency explicit for handoff to simulators.
    """

    buffer_id: str
    layer_id: int
    group_id: int
    buffer_class: str  # R, M, P, E, D
    producer_node_id: str
    consumer_node_ids: List[str]
    start_execution_index: int
    end_execution_index: int
    size_bytes_estimate: float
    size_estimation_method: str
    resident_candidate: bool
    confidence: str
    notes: str = ""


@dataclass
class PipelineGroup:
    """A **contiguous** MoE micro-pipeline within one decoder layer.

    We treat a group as the window over which **intermediate MoE activations may
    plausibly remain resident** in a scratchpad-like structure: routing, packing,
    expert compute, and unpack/reduction collectively form one local dataflow island.

    **Grouping is heuristic:** it uses local structural signals (MoE anchors,
    gap-tolerant inclusion of tiny helpers) rather than a formal PDG.  It is
    **architecture-relevant** because residency policies are usually applied to
    short fused regions, not whole layers.

    **Differs from flat op_profile:** op_profile cannot express that these nodes form
    one adjacent chain; the group id encodes that joint structure for the simulator.
    """

    group_id: int
    layer_id: int
    node_ids: List[str]
    buffer_ids: List[str]
    start_execution_index: int
    end_execution_index: int
    group_type: str
    confidence: str
    notes: str = ""


# ---------------------------------------------------------------------------
# Trace / ordering
# ---------------------------------------------------------------------------

def _trace_first_ts_per_kernel_name(trace_path: Path) -> Dict[str, float]:
    """Map cleaned GPU kernel name → first ``ts`` seen in Chrome trace.

    **Architectural assumption:** the first time a kernel *template* appears in the
    trace is a better proxy for program-side proximity than ``kernel_database.json``
    ordering (which sorts by aggregate duration).  This is still not a perfect
    execution order (streams, async launches), but it anchors MoE stages more
    realistically than duration rank.
    """
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
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
    classified_kernels: List[Dict],
    trace_path: Optional[Path],
) -> Tuple[List[Dict], str, str]:
    """Return classified entries in best-effort **execution-ish** order.

    Returns:
        (ordered_entries, ordering_source, loud_warning_or_empty)

    **Primary:** if ``trace_path`` exists, sort by first kernel timestamp (per cleaned
    kernel name), then by stable ``rank`` / list index.

    **Fallback:** preserve list order from ``kernel_database.json`` and emit a loud
    warning that ordering is weak (DB sorted by duration when generated).
    """
    if not classified_kernels:
        return [], "empty", ""

    if trace_path is not None and Path(trace_path).is_file():
        first_ts = _trace_first_ts_per_kernel_name(Path(trace_path))

        def sort_key(e: Dict) -> Tuple[float, int, str]:
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
        "WARNING: No trace.json — ordered_pipeline uses kernel_database.json list order, "
        "which is typically sorted by aggregate duration (not program order).  "
        "Real scratchpad adjacency may differ; treat dataflow_role / groups as approximate."
    )
    return ordered, "kernel_db_list_fallback", warn


# ---------------------------------------------------------------------------
# dataflow_role inference (context-aware)
# ---------------------------------------------------------------------------

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
class _RoleCtx:
    """Mutable scan state for one layer's pipeline.

    **Why context matters:** ``aten::copy_`` before the first routed expert GEMM is
    treated as **pack_input** (building expert-major layout); the same op after the
    down-proj is **unpack_reduce**.  Generic elementwise inside the expert body is
    **expert_nonlinearity**; elsewhere it stays **other**.
    """

    seen_routing_logits: bool = False
    seen_routing_select: bool = False
    seen_routing_meta: bool = False
    seen_expert_expand: bool = False
    seen_expert_down: bool = False
    in_routing_region: bool = False
    in_expert_body: bool = False


def infer_dataflow_role(
    *,
    op_name: str,
    aten_op_name: str,
    expert_type: str,
    structural_role: str,
    ctx: _RoleCtx,
    is_shared_expert: bool,
) -> str:
    """Map one node to a fine-grained **dataflow_role** using a commented decision tree.

    **Inputs combine** structural_role, expert_type, inferred op_name, and ATen name.
    **Side effect:** updates ``ctx`` for subsequent nodes in the same layer.

    Suggested roles (spec):
      routing_logits, routing_select, routing_metadata,
      pack_input, expert_expand, expert_nonlinearity, expert_down,
      unpack_reduce, shared_expert_expand, shared_expert_down, other
    """
    aten = aten_op_name or ""
    sr = structural_role or ""
    et = expert_type or ""

    # --- Shared / gate expert GEMMs (dimension-tagged by detect) ---
    if et == "shared_expert" and sr == "shared_expert_expand":
        ctx.in_expert_body = True
        return "shared_expert_expand"
    if et == "shared_expert" and sr == "shared_expert_down":
        ctx.seen_expert_down = True
        ctx.in_expert_body = False
        return "shared_expert_down"

    if sr == "moe_gate" or op_name == "moe_gate_proj":
        ctx.seen_routing_logits = True
        ctx.in_routing_region = True
        return "routing_logits"

    if aten in _ROUTING_SELECT_ATEN:
        ctx.seen_routing_select = True
        ctx.in_routing_region = True
        return "routing_select"

    if aten in _ROUTING_META_ATEN:
        ctx.seen_routing_meta = True
        ctx.in_routing_region = True
        return "routing_metadata"

    # Inductor-style MoE may surface as ``aten::_grouped_mm``.  We only treat it as
    # part of the expert MLP chain when ``expert_type`` is already ``routed_expert``.
    # First occurrence in a layer → expand, second → down (heuristic if roles absent).
    if et == "routed_expert" and (
        aten == "aten::_grouped_mm" or "grouped_mm" in aten
    ):
        if not ctx.seen_expert_expand:
            ctx.seen_expert_expand = True
            ctx.in_routing_region = False
            ctx.in_expert_body = True
            return "expert_expand"
        if not ctx.seen_expert_down:
            ctx.seen_expert_down = True
            ctx.in_expert_body = False
            return "expert_down"
        return "expert_down"

    if et == "routed_expert" and sr == "routed_expert_expand":
        ctx.seen_expert_expand = True
        ctx.in_routing_region = False
        ctx.in_expert_body = True
        return "expert_expand"

    if et == "routed_expert" and sr == "routed_expert_down":
        ctx.seen_expert_down = True
        ctx.in_expert_body = False
        return "expert_down"

    if aten in _ACTIVATION_ATEN or op_name == "activation":
        if ctx.in_expert_body or (ctx.seen_expert_expand and not ctx.seen_expert_down):
            return "expert_nonlinearity"
        return "other"

    if aten in _ELEMENTWISE_ATEN:
        if ctx.in_expert_body and ctx.seen_expert_expand and not ctx.seen_expert_down:
            return "expert_nonlinearity"
        return "other"

    if aten in _PACKISH_ATEN or op_name in ("elementwise",):
        # copy_ / index / cat: packing vs unpack depends on GEMM frontier.
        if not ctx.seen_expert_expand and (ctx.in_routing_region or ctx.seen_routing_logits):
            return "pack_input"
        if ctx.seen_expert_down:
            return "unpack_reduce"
        if ctx.seen_expert_expand and not ctx.seen_expert_down:
            # Reshapes between expand and down still touch expert-major activations.
            return "expert_nonlinearity"
        return "other"

    if op_name in ("attn_bmm_kv", "attn_proj") or et == "attention" or sr == "attention":
        return "other"

    if op_name in ("rmsnorm", "layernorm", "softmax"):
        return "other"

    return "other"


# ---------------------------------------------------------------------------
# Node construction (aligned with op_profile expansion)
# ---------------------------------------------------------------------------

def _cta_count(entry: Dict) -> int:
    grid = (entry.get("kernel") or {}).get("grid") or [1, 1, 1]
    cta = 1
    for dim in grid:
        cta *= int(dim) if dim else 1
    return cta


def build_pipeline_nodes(
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    trace_path: Optional[Path] = None,
) -> Tuple[List[PipelineNode], str, str]:
    """Build ordered ``PipelineNode`` list (all layers, layer-major execution order).

    **Execution order:** for each ``layer_id``, we walk ``order_classified_entries``
    and emit nodes in that order — mirroring how a single forward pass visits kernels
    if the ordering source is trustworthy.

    **profile_layer_id:** matches ``op_profile`` semantics (``-1`` for entries that
    are not expanded per layer).  ``layer_id`` on the node is always the layer
    context for this pipeline slot (routed templates are replicated per layer for
    residency analysis).
    """
    ordered, ordering_source, order_note = order_classified_entries(
        classified_kernels, trace_path
    )
    dtype_b = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
    num_layers = max(1, int(num_layers))

    nodes: List[PipelineNode] = []
    global_ei = 0

    for layer_id in range(num_layers):
        layer_ei = 0
        ctx = _RoleCtx()
        for entry in ordered:
            expert_type = entry.get("expert_type", "other")
            structural_role = entry.get("structural_role", "other")
            aten_op = entry.get("aten_op", {})
            aten_op_name = aten_op.get("name", "")
            input_dims = aten_op.get("input_dims", [])
            entry_id = entry.get("id", "")
            stats = entry.get("statistics", {})
            freq = int(stats.get("frequency", 1))
            latency_us = float(stats.get("avg_duration_us", 0.0) or 0.0)

            hbm_fields = _compute_hbm_fields(aten_op_name, input_dims, dtype_b)
            if entry_id in ncu_results:
                ncu = ncu_results[entry_id]
                ncu_hbm = float(
                    (ncu.get("hbm_read_bytes") or 0) + (ncu.get("hbm_write_bytes") or 0)
                )
                if ncu_hbm > 0:
                    hbm_fields = dict(hbm_fields)
                    hbm_fields["hbm_bytes"] = ncu_hbm

            is_shared = expert_type == "shared_expert"
            op_name = _infer_structural_op_name(aten_op_name, structural_role, input_dims)

            ops_count = _ops_per_layer(freq, num_layers) if expert_type != "routed_expert" else 0
            profile_layer_id = layer_id if ops_count > 0 else -1

            def emit_one() -> None:
                nonlocal layer_ei, global_ei, nodes, ctx
                role = infer_dataflow_role(
                    op_name=op_name,
                    aten_op_name=aten_op_name,
                    expert_type=expert_type,
                    structural_role=structural_role,
                    ctx=ctx,
                    is_shared_expert=is_shared,
                )
                node = PipelineNode(
                    node_id=f"L{layer_id}:N{layer_ei}",
                    layer_id=layer_id,
                    execution_index=layer_ei,
                    source_entry_id=str(entry_id) if entry_id else None,
                    op_name=op_name,
                    aten_op_name=aten_op_name,
                    expert_type=expert_type,
                    structural_role=structural_role,
                    dataflow_role=role,
                    activation_bytes=float(hbm_fields.get("activation_bytes", 0.0)),
                    weight_bytes=float(hbm_fields.get("weight_bytes", 0.0)),
                    hbm_bytes=float(hbm_fields.get("hbm_bytes", 0.0)),
                    cta_count=_cta_count(entry),
                    latency_us=latency_us,
                    is_shared_expert=is_shared,
                    profile_layer_id=profile_layer_id,
                    ordering_note=order_note if layer_id == 0 and layer_ei == 0 else "",
                )
                nodes.append(node)
                layer_ei += 1
                global_ei += 1

            if ops_count > 0:
                for _ in range(ops_count):
                    emit_one()
            elif expert_type == "routed_expert":
                # **Architectural choice:** replicate routed template once per layer so
                # simulators can attach residency to layer L even though op_profile uses
                # layer_id=-1 for frequency bookkeeping.
                emit_one()

    return nodes, ordering_source, order_note


def reinfer_dataflow_roles_per_layer(nodes: List[PipelineNode]) -> None:
    """Second pass: reset context per layer so roles depend only on local layer order."""
    by_layer: Dict[int, List[PipelineNode]] = {}
    for n in nodes:
        by_layer.setdefault(n.layer_id, []).append(n)
    for layer_id in sorted(by_layer):
        ctx = _RoleCtx()
        for n in sorted(by_layer[layer_id], key=lambda x: x.execution_index):
            n.dataflow_role = infer_dataflow_role(
                op_name=n.op_name,
                aten_op_name=n.aten_op_name,
                expert_type=n.expert_type,
                structural_role=n.structural_role,
                ctx=ctx,
                is_shared_expert=n.is_shared_expert,
            )


# ---------------------------------------------------------------------------
# Pipeline groups (gap-tolerant)
# ---------------------------------------------------------------------------

def _is_moe_anchor(n: PipelineNode) -> bool:
    if n.expert_type in ("gate", "routed_expert", "shared_expert"):
        return True
    if n.dataflow_role in (
        "routing_logits",
        "routing_select",
        "routing_metadata",
        "pack_input",
        "expert_expand",
        "expert_down",
        "unpack_reduce",
        "shared_expert_expand",
        "shared_expert_down",
    ):
        return True
    return False


def _is_hard_terminator(n: PipelineNode) -> bool:
    # Attention / norm boundaries: outside the local MoE island.
    if n.expert_type == "attention" or n.structural_role == "attention":
        return True
    if n.op_name in ("attn_bmm_kv", "attn_proj", "rmsnorm", "layernorm", "softmax"):
        return True
    return False


def _is_tiny_helper(n: PipelineNode, max_bytes: float = 262144.0) -> bool:
    """Allow a **small** number of generic ops to remain inside a MoE group.

    **Assumption:** tiny metadata or layout helpers (e.g. small copy_) between
    routing and GEMM still belong to the same residency window as the MoE chain.
    """
    return n.dataflow_role == "other" and n.activation_bytes <= max_bytes


def build_moe_pipeline_groups(nodes: List[PipelineNode]) -> List[PipelineGroup]:
    """Partition each layer's nodes into contiguous **PipelineGroup** segments.

    **Heuristic:** start a group when we see a MoE anchor after a terminator / gap.
    Extend through MoE-related nodes; allow up to ``max_other_gap`` tiny ``other``
    nodes between anchors so brief helpers do not fracture the pipeline.

    **Why contiguous:** hardware-controlled scratchpad regions typically cover **short
    fused chains**; this group approximates that window for architecture study.
    """
    max_other_gap = 3
    groups: List[PipelineGroup] = []
    gid = 0

    by_layer: Dict[int, List[PipelineNode]] = {}
    for n in nodes:
        by_layer.setdefault(n.layer_id, []).append(n)

    for layer_id in sorted(by_layer.keys()):
        layer_nodes = sorted(by_layer[layer_id], key=lambda x: x.execution_index)
        i = 0
        while i < len(layer_nodes):
            n = layer_nodes[i]
            if _is_hard_terminator(n):
                i += 1
                continue
            if not _is_moe_anchor(n):
                i += 1
                continue

            buf: List[PipelineNode] = []
            other_streak = 0
            while i < len(layer_nodes):
                cur = layer_nodes[i]
                if _is_hard_terminator(cur) and buf:
                    break
                if _is_moe_anchor(cur):
                    buf.append(cur)
                    other_streak = 0
                    i += 1
                    continue
                if cur.dataflow_role == "other" and _is_tiny_helper(cur):
                    if other_streak < max_other_gap:
                        buf.append(cur)
                        other_streak += 1
                        i += 1
                        continue
                    break
                if cur.dataflow_role == "other":
                    break
                buf.append(cur)
                i += 1

            if not buf:
                continue

            for nn in buf:
                nn.pipeline_group_id = gid

            conf, gnotes = _group_confidence(buf)
            g = PipelineGroup(
                group_id=gid,
                layer_id=layer_id,
                node_ids=[nn.node_id for nn in buf],
                buffer_ids=[],
                start_execution_index=buf[0].execution_index,
                end_execution_index=buf[-1].execution_index,
                group_type="moe_decode_subpipeline",
                confidence=conf,
                notes=gnotes,
            )
            groups.append(g)
            gid += 1

    return groups


def _group_confidence(buf: Sequence[PipelineNode]) -> Tuple[str, str]:
    roles = {n.dataflow_role for n in buf}
    needed = {"routing_logits", "expert_expand", "expert_down"}
    if needed.issubset(roles) and ("pack_input" in roles or "routing_metadata" in roles):
        return "high", "Full MoE role set present in contiguous window."
    if "expert_expand" in roles and "expert_down" in roles:
        return "medium", "Expert GEMM bookends present; routing/pack roles partial or merged."
    if _is_moe_anchor(buf[0]):
        return "low", "MoE anchor without clear expert_expand/down sequence."
    return "low", "Weak MoE pattern."


# ---------------------------------------------------------------------------
# Logical buffers R/M/P/E/D
# ---------------------------------------------------------------------------

def _max_act(nodes: Iterable[PipelineNode]) -> float:
    return max((n.activation_bytes for n in nodes), default=0.0)


def _first_node(nodes: Iterable[PipelineNode]) -> Optional[PipelineNode]:
    lst = list(nodes)
    if not lst:
        return None
    return min(lst, key=lambda x: x.execution_index)


def reconstruct_logical_buffers(
    groups: List[PipelineGroup],
    nodes: List[PipelineNode],
) -> List[LogicalBuffer]:
    """Reconstruct R/M/P/E/D buffers per group from **dataflow_role** subgraphs.

    **R** — routing selection outputs (logits / scores).
    **M** — routing metadata (indices, counts, offsets).
    **P** — packed expert-major input activations.
    **E** — expert intermediate (post-expand, incl. nonlinearity updates).
    **D** — packed expert outputs before scatter / reduction.

    Sizes use **max activation_bytes** among nodes mapped to that stage, explicit
    and conservative when multiple kernels touch the same logical tensor.

    **These buffers are not in the trace**; they exist so the simulator can model
    **live ranges** instead of summing isolated operator footprints.
    """
    node_map = {n.node_id: n for n in nodes}
    buffers: List[LogicalBuffer] = []
    bid = 0

    for g in groups:
        gn = [node_map[nid] for nid in g.node_ids if nid in node_map]
        if not gn:
            continue

        def nodes_with_roles(roles: frozenset) -> List[PipelineNode]:
            return [n for n in gn if n.dataflow_role in roles]

        r_nodes = nodes_with_roles(frozenset({"routing_logits", "routing_select"}))
        m_nodes = nodes_with_roles(frozenset({"routing_metadata"}))
        p_nodes = nodes_with_roles(frozenset({"pack_input"}))
        e_nodes = nodes_with_roles(
            frozenset({"expert_expand", "expert_nonlinearity", "shared_expert_expand"})
        )
        d_nodes = nodes_with_roles(frozenset({"expert_down", "shared_expert_down"}))
        u_nodes = nodes_with_roles(frozenset({"unpack_reduce"}))

        def make_buf(
            cls: str,
            producers: List[PipelineNode],
            consumers: List[PipelineNode],
            resident: bool,
            conf: str,
            note: str,
        ) -> Optional[LogicalBuffer]:
            nonlocal bid
            if not producers and not consumers:
                return None
            # A logical buffer is anchored on a **producer stage**; consumers may be
            # absent only for dead code in the trace (then we still emit for audit).
            if not producers:
                return None
            prod = _first_node(producers)
            if prod is None:
                return None
            cons_list = [n for n in consumers if n is not None]
            all_n = producers + cons_list
            start = min(n.execution_index for n in all_n)
            end = max(n.execution_index for n in all_n)
            size = _max_act(producers) if producers else _max_act(cons_list)
            if size <= 0 and cons_list:
                size = _max_act(cons_list)
            buf = LogicalBuffer(
                buffer_id=f"G{g.group_id}:B{bid}",
                layer_id=g.layer_id,
                group_id=g.group_id,
                buffer_class=cls,
                producer_node_id=prod.node_id,
                consumer_node_ids=[n.node_id for n in cons_list],
                start_execution_index=start,
                end_execution_index=end,
                size_bytes_estimate=float(size),
                size_estimation_method="max_activation_bytes_among_role_nodes",
                resident_candidate=resident,
                confidence=conf,
                notes=note,
            )
            bid += 1
            return buf

        conf = g.confidence

        # R: produced by logits/select; consumed by metadata, pack, unpack.
        r_cons = m_nodes + p_nodes + u_nodes
        b_r = make_buf(
            "R",
            r_nodes,
            r_cons,
            resident=True,
            conf=conf,
            note="Logical routing tensor; small vs P/E/D but carried for completeness.",
        )
        if b_r:
            buffers.append(b_r)

        b_m = make_buf(
            "M",
            m_nodes,
            p_nodes + u_nodes,
            True,
            conf,
            "Metadata consumed when packing inputs and when reducing outputs.",
        )
        if b_m:
            buffers.append(b_m)

        b_p = make_buf(
            "P",
            p_nodes,
            e_nodes,
            True,
            conf,
            "Packed expert-major activations feeding expert GEMMs.",
        )
        if b_p:
            buffers.append(b_p)

        b_e = make_buf(
            "E",
            e_nodes,
            d_nodes,
            True,
            conf,
            "Expert MLP intermediate; nonlinearity updates assumed in-place on E.",
        )
        if b_e:
            buffers.append(b_e)

        b_d = make_buf(
            "D",
            d_nodes,
            u_nodes,
            True,
            conf,
            "Expert outputs prior to scatter / reduction back to token layout.",
        )
        if b_d:
            buffers.append(b_d)

    g_buf: Dict[int, List[str]] = {}
    for b in buffers:
        g_buf.setdefault(b.group_id, []).append(b.buffer_id)
    for g in groups:
        g.buffer_ids = g_buf.get(g.group_id, [])

    return buffers


# ---------------------------------------------------------------------------
# Live range & workspace
# ---------------------------------------------------------------------------

def estimate_group_live_buffers(
    group: PipelineGroup,
    buffers: Sequence[LogicalBuffer],
) -> Dict[str, Any]:
    """Summarize which buffers are live across the group's execution index range.

    **Not a dynamic liveness analysis** — we use structural stage roles to approximate
    intervals for documentation and quick simulator ingestion.
    """
    mine = [b for b in buffers if b.group_id == group.group_id]
    return {
        "group_id": group.group_id,
        "layer_id": group.layer_id,
        "buffers": [
            {
                "buffer_id": b.buffer_id,
                "class": b.buffer_class,
                "interval": [b.start_execution_index, b.end_execution_index],
                "size_bytes_estimate": b.size_bytes_estimate,
            }
            for b in mine
        ],
    }


def estimate_peak_workspace_bytes(
    group: PipelineGroup,
    buffers: Sequence[LogicalBuffer],
    overlap_mode: str = "staged",
) -> Dict[str, Any]:
    """Bound transient scratchpad needs for one MoE group.

    **Staged model:** assume P, E, D are **staged** so peak ≈ R + M + max(P, E, D).

    **Conservative overlap:** peak ≈ R + M + max(P+E, E+D, P+D) — assumes adjacent
    stage buffers may briefly coexist (e.g. ping-pong or partial overlap).

    These are **intentional upper/structured bounds** for architecture storytelling,
    not cycle-accurate allocation sizes.
    """
    mine = {b.buffer_class: b for b in buffers if b.group_id == group.group_id}
    R = mine.get("R")
    M = mine.get("M")
    P = mine.get("P")
    E = mine.get("E")
    D = mine.get("D")

    def sz(b: Optional[LogicalBuffer]) -> float:
        return float(b.size_bytes_estimate) if b else 0.0

    r_v, m_v = sz(R), sz(M)
    p_v, e_v, d_v = sz(P), sz(E), sz(D)

    if overlap_mode == "conservative":
        peak = r_v + m_v + max(p_v + e_v, e_v + d_v, p_v + d_v)
        formula = "R + M + max(P+E, E+D, P+D)"
    else:
        peak = r_v + m_v + max(p_v, e_v, d_v)
        formula = "R + M + max(P, E, D)"

    return {
        "group_id": group.group_id,
        "layer_id": group.layer_id,
        "overlap_mode": overlap_mode,
        "formula": formula,
        "peak_workspace_bytes_estimate": peak,
        "components": {"R": r_v, "M": m_v, "P": p_v, "E": e_v, "D": d_v},
    }


# ---------------------------------------------------------------------------
# Enrich flat op_profile records
# ---------------------------------------------------------------------------

def enrich_op_profile_records(
    records: List[Dict],
    nodes: List[PipelineNode],
) -> None:
    """Attach optional **advisory** dataflow fields to op_profile rows (in place).

    **Backward compatibility:** existing consumers ignore unknown keys.  These fields
    are derived from the ordered pipeline pass and may be null when no match exists.

    **Matching (best-effort):** primary queue is ``(layer_id, source_entry_id)`` in
    global execution order.  Shared-expert template reconstruction rewrites ``op_name``
    but preserves ``source_entry_id`` on clones, so consuming nodes in emission order
    aligns gate/up/down rows with successive matching kernels when the DB used
    distinct entry IDs.  Fallback: ``(layer_id, op_name)``.  Routed rows with
    ``layer_id=-1`` attach a **layer-0 advisory** node only (one profile row models
    all layers).
    """
    from collections import defaultdict, deque

    ordered = sorted(nodes, key=lambda x: (x.layer_id, x.execution_index))
    q_layer_eid = defaultdict(deque)
    q_layer_op = defaultdict(deque)
    for n in ordered:
        q_layer_eid[(n.layer_id, n.source_entry_id or "")].append(n)
        q_layer_op[(n.layer_id, n.op_name)].append(n)

    logical_produced: Dict[str, str] = {
        "routing_logits": "R",
        "routing_select": "R",
        "routing_metadata": "M",
        "pack_input": "P",
        "expert_expand": "E",
        "expert_nonlinearity": "E",
        "shared_expert_expand": "E",
        "expert_down": "D",
        "shared_expert_down": "D",
        "unpack_reduce": "D",
    }

    logical_consumed: Dict[str, List[str]] = {
        "routing_metadata": ["R"],
        "pack_input": ["R", "M"],
        "expert_expand": ["P"],
        "expert_nonlinearity": ["E"],
        "expert_down": ["E"],
        "unpack_reduce": ["D", "M", "R"],
        "shared_expert_expand": ["P"],
        "shared_expert_down": ["E"],
    }

    for r in records:
        lid = int(r.get("layer_id", -1))
        eid = str(r.get("source_entry_id") or "")
        opn = str(r.get("op_name") or "")
        n = None
        if lid >= 0:
            if q_layer_eid[(lid, eid)]:
                n = q_layer_eid[(lid, eid)].popleft()
            elif q_layer_op[(lid, opn)]:
                n = q_layer_op[(lid, opn)].popleft()
        else:
            c0 = [
                x
                for x in ordered
                if (x.source_entry_id or "") == eid and x.op_name == opn and x.layer_id == 0
            ]
            n = c0[0] if c0 else None

        r["execution_index"] = n.execution_index if n else None
        r["dataflow_role"] = n.dataflow_role if n else None
        r["pipeline_group_id"] = n.pipeline_group_id if n else None
        role = n.dataflow_role if n else None
        r["logical_buffer_produced"] = logical_produced.get(role) if role else None
        r["logical_buffers_consumed"] = logical_consumed.get(role) if role else None
        r["producer_consumer_notes"] = (
            "Derived from ordered dataflow pipeline; approximate when trace order unknown."
            if n
            else None
        )


# ---------------------------------------------------------------------------
# JSON + debug emission
# ---------------------------------------------------------------------------

def _node_to_dict(n: PipelineNode) -> Dict[str, Any]:
    return asdict(n)


def emit_dataflow_artifacts(
    *,
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str,
    ncu_results: Optional[Dict[str, Dict]],
    trace_path: Optional[Path],
    output_dir: Path,
    moe_debug_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write ``op_pipeline.json``, ``dataflow_profile.json``, and ``dataflow.debug.txt``.

    Returns a dict with string keys ``op_pipeline``, ``dataflow_profile``,
    ``dataflow_debug`` mapping to paths, plus ``nodes`` (list of ``PipelineNode``)
    for optional enrichment of ``op_profile.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes, ordering_source, order_note = build_pipeline_nodes(
        classified_kernels,
        num_layers=num_layers,
        precision=precision,
        ncu_results=ncu_results,
        trace_path=trace_path,
    )
    reinfer_dataflow_roles_per_layer(nodes)

    groups = build_moe_pipeline_groups(nodes)
    buffers = reconstruct_logical_buffers(groups, nodes)

    workspace_staged = [
        estimate_peak_workspace_bytes(g, buffers, overlap_mode="staged") for g in groups
    ]
    workspace_conservative = [
        estimate_peak_workspace_bytes(g, buffers, overlap_mode="conservative")
        for g in groups
    ]

    group_live = [estimate_group_live_buffers(g, buffers) for g in groups]

    meta = {
        "ordering_source": ordering_source,
        "ordering_warning": order_note,
        "num_layers": num_layers,
        "precision": precision,
        "trace_path_used": str(trace_path) if trace_path and Path(trace_path).is_file() else None,
        "confidence_note": (
            "Confidence labels describe reconstruction fidelity of the buffer graph, "
            "not expected model performance."
        ),
        "simulator_handoff": (
            "Use dataflow_profile.json for scratchpad / controlled-L2 residency modeling; "
            "op_profile.json remains a flat aggregate sorted by op_name."
        ),
    }

    op_pipeline_path = output_dir / "op_pipeline.json"
    op_pipeline_path.write_text(
        json.dumps(
            {"metadata": meta, "nodes": [_node_to_dict(n) for n in nodes]},
            indent=2,
        ),
        encoding="utf-8",
    )

    dataflow_path = output_dir / "dataflow_profile.json"
    payload = {
        "metadata": meta,
        "nodes": [_node_to_dict(n) for n in nodes],
        "groups": [asdict(g) for g in groups],
        "buffers": [asdict(b) for b in buffers],
        "group_workspace_estimates": {
            "staged": workspace_staged,
            "conservative": workspace_conservative,
            "group_live_buffers": group_live,
        },
    }
    dataflow_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dbg_path = output_dir / "dataflow.debug.txt"
    lines = [
        "# dataflow.debug.txt — architectural reconstruction audit",
        "# Purpose: make producer/consumer inference and grouping legible to humans.",
        "",
        f"ordering_source={ordering_source}",
        f"note={order_note}",
        "",
        "--- nodes per layer (execution_index, role, op, entry) ---",
    ]
    by_layer: Dict[int, List[PipelineNode]] = {}
    for n in nodes:
        by_layer.setdefault(n.layer_id, []).append(n)
    for lid in sorted(by_layer):
        lines.append(f"layer {lid}:")
        for n in sorted(by_layer[lid], key=lambda x: x.execution_index):
            lines.append(
                f"  ei={n.execution_index} role={n.dataflow_role} op={n.op_name} "
                f"aten={n.aten_op_name} entry={n.source_entry_id} group={n.pipeline_group_id}"
            )
    lines.append("")
    lines.append("--- pipeline groups ---")
    for g in groups:
        lines.append(
            f"group {g.group_id} layer={g.layer_id} ei=[{g.start_execution_index},{g.end_execution_index}] "
            f"conf={g.confidence} nodes={len(g.node_ids)}"
        )
    lines.append("")
    lines.append("--- logical buffers ---")
    for b in buffers:
        lines.append(
            f"{b.buffer_id} class={b.buffer_class} group={b.group_id} "
            f"prod={b.producer_node_id} cons={b.consumer_node_ids} "
            f"size={b.size_bytes_estimate:.0f} conf={b.confidence}"
        )
    lines.append("")
    lines.append("--- workspace staged ---")
    for w in workspace_staged:
        lines.append(
            f"group {w['group_id']}: peak={w['peak_workspace_bytes_estimate']:.0f} "
            f"({w['formula']})"
        )
    dbg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    append_moe_op_profile_debug(
        moe_debug_log_path,
        f"[moe.dataflow] wrote {op_pipeline_path.name}, {dataflow_path.name}, {dbg_path.name} "
        f"ordering_source={ordering_source}",
    )

    return {
        "op_pipeline": op_pipeline_path,
        "dataflow_profile": dataflow_path,
        "dataflow_debug": dbg_path,
        "nodes": nodes,
    }


def resolve_trace_path(kernel_db_path: Optional[Path]) -> Optional[Path]:
    """Default trace.json location next to ``kernel_database.json``."""
    if kernel_db_path is None:
        return None
    p = Path(kernel_db_path).parent / "trace.json"
    return p if p.is_file() else None


def resolve_trace_path_for_op_profile(output_path: Optional[Path]) -> Optional[Path]:
    """Best-effort ``trace.json`` next to the MoE experiment directory.

    When ``op_profile.json`` lives under ``.../moe_profile/``, look for a sibling
    of that folder (``.../trace.json``).  Otherwise try beside the JSON file.
    """
    if output_path is None:
        return None
    parent = Path(output_path).parent
    candidates = []
    if parent.name == "moe_profile":
        candidates.append(parent.parent / "trace.json")
    candidates.append(parent / "trace.json")
    for c in candidates:
        if c.is_file():
            return c
    return None
