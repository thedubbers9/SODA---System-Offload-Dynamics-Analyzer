# Abstraction: canonical **simulator-facing** records for MoE scratchpad / residency
# modeling.  This module defines *what* we export, not *how* we infer it.
#
# Differs from the flat op-profile path: op_profile.json aggregates by op name and
# sorts for human diffs — it cannot represent ordered producer-consumer chains,
# logical buffers R/M/P/E/D, or group-local workspace bounds.
#
# **Experimental:** optimized for architecture-facing dataflow reconstruction only;
# schemas may evolve without preserving legacy consumers.
"""Dataclasses for the experimental MoE local dataflow graph.

Every type below is sized for **scratchpad-style reuse simulation**: execution order,
local MoE windows, reconstructed logical tensors, and explicit buffer edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KernelNode:
    """One **ordered** kernel-database template occurrence in a per-layer stream.

    Architectural entity: a single slot in the best-effort program order we recover
    from ``kernel_database.json`` plus optional ``trace.json``.  This is still one
    *template* (unique op/kernel/shape row), not every dynamic GPU invocation —
    SODA does not materialize per-invocation rows.

    **Why residency modeling needs it:** scratchpad policies depend on **which
    kernels are adjacent in time** and how big their reported activations are.  A
    flat per-op aggregate destroys that adjacency.

    **Differs from op_profile:** op_profile rows are sorted by ``op_name`` and may
    merge many logical slots; ``KernelNode`` preserves ``execution_index`` and ties
    back to ``source_entry_id`` and trace timestamps for schedule fidelity.
    """

    node_id: str
    execution_index: int
    trace_ts_us: Optional[float]
    layer_id: int
    source_entry_id: Optional[str]
    kernel_name: str
    aten_op_name: str
    op_name: str
    input_dims: Any
    expert_type: str
    structural_role: str
    activation_bytes: float
    weight_bytes: float
    hbm_bytes: float
    latency_us: float
    cta_count: int
    is_shared_expert: bool
    notes: str = ""
    # Grouped-GEMM semantics (``aten::_grouped_mm``): expert axis is explicit in weights.
    grouped_mm_num_experts: Optional[int] = None
    grouped_mm_input_dim: Optional[int] = None
    grouped_mm_output_dim: Optional[int] = None
    pipeline_group_id: Optional[int] = None


@dataclass
class PipelineGroup:
    """One **local MoE subgraph** candidate: a short execution window around anchors.

    Architectural entity: the fused or near-fused region where routing, packing,
    expert MLP (often ``_grouped_mm`` chains), and unpack/reduction plausibly share
    a controlled on-chip workspace.  Hardware does not see “the whole decoder
    layer” as one residency unit — it sees **local micro-pipelines**.

    **Why needed:** simulators allocate scratchpad/L2 budget to **groups** of ops
    with overlapping logical buffers, not to isolated kernels.

    **Differs from op_profile:** op_profile cannot express ``start_execution_index``
    … ``end_execution_index`` windows or membership lists; this type does.
    """

    group_id: int
    layer_id: int
    start_execution_index: int
    end_execution_index: int
    node_ids: List[str]
    group_kind: str
    confidence: str
    notes: str = ""


@dataclass
class StageNode:
    """A ``KernelNode`` enriched with **local MoE stage** semantics inside one group.

    Architectural entity: the same physical kernel slot as ``KernelNode``, plus the
    **interpreted** role (routing vs pack vs expert body vs unpack) within an
    anchor-defined window.  Generic ATen ops like ``copy_`` only gain meaning
    **relative to nearby grouped expert GEMMs** — that interpretation lives here.

    **Why needed:** the simulator reasons about **stages** (R→M→P→E→D), not raw
    ATen names.  Predecessor/successor links summarize the implied linear chain
    within the group for quick graph walks.

    **Differs from op_profile:** op_profile has no ``stage_role`` or explicit
    per-group adjacency; those are reconstructed here for dataflow export only.
    """

    node_id: str
    group_id: int
    stage_role: str
    stage_confidence: str
    predecessor_node_ids: List[str] = field(default_factory=list)
    successor_node_ids: List[str] = field(default_factory=list)
    # Snapshot of kernel fields most simulators need (avoid indirection-only JSON).
    kernel: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicalBuffer:
    """A **reconstructed** logical intermediate tensor (R, M, P, E, or D).

    Architectural entity: R (routing), M (metadata), P (packed expert-major inputs),
    E (expert MLP intermediate), D (packed expert outputs before scatter/reduce).
    These tensors **do not appear as explicit rows** in the kernel DB; we infer
    them so a simulator can attach **sizes and live intervals** to conceptual
    activations instead of summing unrelated op bytes.

    **Why needed:** residency / scratchpad simulation is about **buffer lifetimes**
    across kernels.  Multiple kernels may read/write the same logical buffer; this
    record names that buffer once and lists producer/consumer **node_ids**.

    **Differs from op_profile:** op_profile reports per-op bytes; it does not
    connect producers to consumers or classify buffers into R/M/P/E/D.
    """

    buffer_id: str
    group_id: int
    layer_id: int
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
class WorkspaceEstimate:
    """Group-local **architecture-facing** workspace bounds (not exact allocators).

    Architectural entity: a pair of conservative scalars describing how much
    transient state might need to be **simultaneously resident** if P/E/D are
    staged vs if adjacent stage buffers overlap (double-buffer, partial lifetime
    overlap, etc.).

    **Why needed:** scratchpad sizing studies need **R+M+max(P,E,D)** vs a more
    conservative **R+M+max(P+E,E+D,P+D)** style bound — both are explicit here.

    **Differs from op_profile:** op_profile has no notion of grouped peak live
    sets; those formulas are defined only on the logical buffer graph.
    """

    group_id: int
    layer_id: int
    R_bytes: float
    M_bytes: float
    P_bytes: float
    E_bytes: float
    D_bytes: float
    peak_bytes_staged: float
    peak_bytes_conservative: float
    notes: str = ""
