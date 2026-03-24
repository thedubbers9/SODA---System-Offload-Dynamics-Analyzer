"""Compact debug sections for minimal MoE routed-expert reconstruction.

Default output is small (Sections A–D). Per-GEMM shape lines require
``debug_full_layer=True`` (CLI: ``--debug-full-layer``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from soda.moe.moe_dataflow.buffers import ChainBuffers
from soda.moe.moe_dataflow.ordering import StreamNode
from soda.moe.moe_dataflow.pairing import PairingResult
from soda.moe.moe_dataflow.windows import ChainWindow


def render_debug(
    *,
    ordering_source: str,
    order_note: str,
    trace_path_used: Optional[str],
    section_a: List[str],
    section_b: List[str],
    section_c: List[str],
    section_d: List[str],
) -> str:
    lines = [
        "# moe_dataflow.debug.txt — minimal routed-expert MoE (anchor-first)",
        "# No legacy layer-wide groups; no shared-expert merge in this path.",
        "",
        "## Ordering",
        f"source: {ordering_source}",
        order_note or "(no note)",
        f"trace_path: {trace_path_used or '(none)'}",
        "",
        "## Section A — Candidate gates (moe_gate_proj)",
    ]
    lines.extend(section_a or ["(empty)"])
    lines.extend(
        [
            "",
            "## Section B — Routed-expert windows",
        ]
    )
    lines.extend(section_b or ["(empty)"])
    lines.extend(
        [
            "",
            "## Section C — Grouped GEMM pairs",
        ]
    )
    lines.extend(section_c or ["(empty)"])
    lines.extend(
        [
            "",
            "## Section D — Logical buffers",
        ]
    )
    lines.extend(section_d or ["(empty)"])
    lines.append("")
    return "\n".join(lines)


def format_section_a_candidate_gates(
    layer_id: int,
    candidates: List[int],
    decisions: Dict[int, Tuple[bool, str]],
    *,
    single_layer_mode: bool,
) -> List[str]:
    """Per gate: L{layer} ei=… accepted|rejected + reason."""
    if not single_layer_mode and layer_id != 0:
        acc = sum(1 for ei in candidates if decisions.get(ei, (False, ""))[0])
        return [
            f"L{layer_id}: {len(candidates)} gate candidate(s), {acc} accepted "
            f"(use --layer {layer_id} for per-gate reasons)"
        ]
    out: List[str] = []
    for ei in candidates:
        ok, reason = decisions.get(ei, (False, "?"))
        status = "accepted" if ok else "rejected"
        out.append(f"L{layer_id} ei={ei} {status}: {reason}")
    if not candidates:
        out.append(f"L{layer_id}: (no moe_gate_proj in stream)")
    return out


def _role_lines(pr: PairingResult) -> List[str]:
    logits = [c.execution_index for c in pr.classified if c.coarse_class == "routing_logits"]
    sel = [c.execution_index for c in pr.classified if c.coarse_class == "routing_select"]
    meta = [c.execution_index for c in pr.classified if c.coarse_class == "routing_metadata"]
    gmm = [c.execution_index for c in pr.classified if c.coarse_class == "grouped_expert_gemm"]
    post = [c.execution_index for c in pr.classified if c.coarse_class == "post_expert_candidate"]
    unk = [c.execution_index for c in pr.classified if c.coarse_class == "unknown_within_window"]
    lines = [
        f"  routing_logits: {logits}",
        f"  routing_select: {sel}",
        f"  routing_metadata: {meta}",
        f"  grouped_expert_gemm: {gmm}",
    ]
    if post:
        lines.append(f"  post_expert_candidate: {post}")
    if unk:
        lines.append(f"  unknown_within_window: {unk}")
    return lines


def format_section_b_routed_windows(
    layer_id: int,
    anchor_ei: int,
    cw: ChainWindow,
    pr: PairingResult,
    layer_ops: List[StreamNode],
    *,
    debug_full_layer: bool,
) -> List[str]:
    lines = [
        f"L{layer_id} anchor_ei={anchor_ei} window=[{cw.start_ei},{cw.end_ei}]",
        "  Roles:",
    ]
    lines.extend(_role_lines(pr))
    if pr.odd_gemm_warning:
        lines.append("  WARNING: odd grouped GEMM count — last one dropped from pairing")
    if debug_full_layer:
        lines.append("  --- grouped GEMM shape detail (ei, entry, H, R, T, pair, label) ---")
        lines.extend(enrich_gemm_debug_with_shapes(layer_ops, pr))
    return lines


def format_section_c_gemm_pairs(
    layer_id: int,
    anchor_ei: int,
    pr: PairingResult,
    layer_ops: List[StreamNode],
    *,
    debug_full_layer: bool,
) -> List[str]:
    lines: List[str] = [f"L{layer_id} anchor_ei={anchor_ei}"]
    for p in pr.pairs:
        g0 = layer_ops[p.gemm0_ei]
        g1 = layer_ops[p.gemm1_ei]
        shape_bit = ""
        if debug_full_layer:
            shape_bit = (
                f"  shapes: gemm0 T={g0.grouped_mm_T} H={g0.grouped_mm_H} R={g0.grouped_mm_R} | "
                f"gemm1 T={g1.grouped_mm_T} R={g1.grouped_mm_R}"
            )
        lines.append(f"  pair{p.pair_id}: ({p.gemm0_ei},{p.gemm1_ei}){shape_bit}")
    if not pr.pairs:
        lines.append("  (no pairs)")
    return lines


def format_section_d_buffers(cb: ChainBuffers) -> List[str]:
    names = [str(b.get("name", "?")) for b in cb.buffers]
    line = " ".join(names) if names else "(none)"
    return [f"L{cb.layer_id} anchor_ei={cb.anchor_ei}: {line}"]


def enrich_gemm_debug_with_shapes(layer_ops: List[StreamNode], pr: PairingResult) -> List[str]:
    lines: List[str] = []
    for c in pr.classified:
        if c.coarse_class != "grouped_expert_gemm":
            continue
        n = layer_ops[c.execution_index]
        if c.pair_gemm_slot == "gemm0":
            alias = "pair_expand"
        elif c.pair_gemm_slot == "gemm1":
            alias = "pair_down"
        else:
            alias = "n/a"
        lines.append(
            f"    ei={c.execution_index} entry={c.kernel_entry_name} "
            f"H={n.grouped_mm_H} R={n.grouped_mm_R} T={n.grouped_mm_T} "
            f"pair_id={c.gemm_pair_index} label={c.pair_role_label} ({alias})"
        )
    return lines
