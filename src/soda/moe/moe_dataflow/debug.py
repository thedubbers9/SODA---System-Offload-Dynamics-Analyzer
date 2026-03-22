"""Compact human-auditable debug output for minimal MoE chains.

Organized as Section A (anchors), B (chains), C (buffers). Optional ``focus_layer``
limits verbose detail to one layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from soda.moe.moe_dataflow.buffers import ChainBuffers
from soda.moe.moe_dataflow.ordering import StreamNode
from soda.moe.moe_dataflow.pairing import PairingResult
from soda.moe.moe_dataflow.windows import ChainWindow


def render_debug(
    *,
    ordering_source: str,
    order_note: str,
    trace_path_used: Optional[str],
    anchor_section: List[str],
    chain_sections: List[str],
    buffer_sections: List[str],
) -> str:
    lines = [
        "# moe_dataflow.debug.txt — minimal MoE routed-expert chain reconstruction",
        "# Scope: anchor-validated gate → select/metadata → paired _grouped_mm only.",
        "# Not a full graph; tensor identity is not recovered.",
        "",
        "## Ordering",
        f"source: {ordering_source}",
        order_note or "(no note)",
        f"trace_path: {trace_path_used or '(none)'}",
        "",
    ]
    lines.append("## Section A — Gate anchors (per layer)")
    lines.extend(anchor_section or ["(empty)"])
    lines.append("")
    lines.append("## Section B — Minimal chains")
    lines.extend(chain_sections or ["(empty)"])
    lines.append("")
    lines.append("## Section C — Logical buffers")
    lines.extend(buffer_sections or ["(empty)"])
    lines.append("")
    return "\n".join(lines)


def format_anchor_lines(
    layer_id: int,
    candidates: List[int],
    decisions: Dict[int, tuple[bool, str]],
    *,
    focus_layer: Optional[int],
) -> List[str]:
    """One layer block for Section A."""
    if focus_layer is not None and layer_id != focus_layer:
        acc = sum(1 for ei in candidates if decisions.get(ei, (False, ""))[0])
        return [
            f"Layer {layer_id}: {len(candidates)} moe_gate_proj candidate(s), {acc} accepted "
            f"(use --layer {layer_id} for per-anchor reasons)"
        ]
    head = [f"Layer {layer_id} — candidate gate (moe_gate_proj) eis: {candidates or '(none)'}"]
    for ei in candidates:
        ok, reason = decisions.get(ei, (False, "?"))
        head.append(f"  ei={ei}: {'ACCEPT' if ok else 'REJECT'} — {reason}")
    if not candidates:
        head.append("  (no moe_gate_proj in stream)")
    return head


def format_chain_lines(
    layer_id: int,
    anchor_ei: int,
    cw: ChainWindow,
    pr: PairingResult,
    layer_ops: List[StreamNode],
    *,
    focus_layer: Optional[int],
) -> List[str]:
    sel = [c.execution_index for c in pr.classified if c.coarse_class == "routing_select"]
    meta = [c.execution_index for c in pr.classified if c.coarse_class == "routing_metadata"]
    gmm = list(pr.grouped_mm_eis)
    pairs_str = ", ".join(f"({p.expand_ei},{p.down_ei})" for p in pr.pairs)
    lines = [
        f"Layer {layer_id} anchor_ei={anchor_ei} window [{cw.start_ei},{cw.end_ei}]",
        f"  select eis: {sel}",
        f"  metadata eis: {meta}",
        f"  grouped GEMMs: {gmm}",
        f"  pairs: {pairs_str}",
    ]
    if pr.odd_gemm_warning:
        lines.append("  WARNING: odd count of grouped GEMMs — last one dropped from pairing")
    if focus_layer is None or layer_id == focus_layer:
        lines.append("  --- grouped GEMM shape summary (ei, entry, H, R, T, pair, logical role) ---")
        lines.extend(enrich_gemm_debug_with_shapes(layer_ops, pr))
    return lines


def format_buffer_lines(cb: ChainBuffers, *, focus_layer: Optional[int]) -> List[str]:
    if focus_layer is not None and cb.layer_id != focus_layer:
        return [f"Layer {cb.layer_id} anchor {cb.anchor_ei}: {len(cb.buffers)} buffers (see JSON)"]
    lines = [
        f"Layer {cb.layer_id} anchor_ei={cb.anchor_ei}",
        "  buffers:",
    ]
    for b in cb.buffers:
        lines.append(f"    {b.get('name')}: class={b.get('class')} bytes≈{b.get('size_bytes_estimate', 0):.0f}")
    lines.append("  shape / pair sizing:")
    for row in cb.shape_debug:
        lines.append(
            f"    pair {row.get('pair_id')}: expand_ei={row.get('expand_ei')} down_ei={row.get('down_ei')} "
            f"T={row.get('T_expand')}/{row.get('T_down')} H={row.get('H')} "
            f"R={row.get('R_expand')}/{row.get('R_down')} uncertain={row.get('uncertain_sizing')}"
        )
    return lines


def enrich_gemm_debug_with_shapes(layer_ops: List[StreamNode], pr: PairingResult) -> List[str]:
    """Append H,R,T from stream nodes for Section B detail."""
    lines = []
    for c in pr.classified:
        if c.coarse_class != "grouped_expert_gemm":
            continue
        n = layer_ops[c.execution_index]
        lines.append(
            f"    ei={c.execution_index} entry={c.kernel_entry_name} "
            f"H={n.grouped_mm_H} R={n.grouped_mm_R} T={n.grouped_mm_T} "
            f"pair_id={c.gemm_pair_index} logical={c.gemm_logical_role}"
        )
    return lines
