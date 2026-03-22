"""CLI and orchestration for minimal MoE routed-expert chain reconstruction.

This is a minimal MoE-local reconstruction pass: it does not reconstruct the full
graph, infer exact tensor identity, or integrate shared experts. It exists only
for architectural intermediate-residency modeling (logical R/M/P/E/D buffers).

Usage::

    python -m soda.moe.moe_dataflow.main --kernel-db ... --out-dir ... [--trace ...] [--layer N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.moe.dataflow import resolve_trace_path
from soda.moe.moe_dataflow.anchors import (
    ANCHOR_VALIDATION_FORWARD_OPS,
    find_gate_candidates,
    validate_gate_anchor,
)
from soda.moe.moe_dataflow.buffers import (
    all_buffers_to_json,
    all_chains_to_json,
    build_chain_buffers,
)
from soda.moe.moe_dataflow.debug import (
    format_anchor_lines,
    format_buffer_lines,
    format_chain_lines,
    render_debug,
)
from soda.moe.moe_dataflow.loader import classify_kernels_from_db, load_kernel_database
from soda.moe.moe_dataflow.ordering import build_ordered_stream_per_layer, order_classified_entries
from soda.moe.moe_dataflow.pairing import (
    PairingResult,
    analyze_chain_window,
    pairing_result_to_json_dict,
)
from soda.moe.moe_dataflow.windows import build_chain_window
from soda.moe.op_profile import _detect_num_layers


def _resolve_num_layers(
    kernel_db: Dict[str, Any],
    classified: List[Dict[str, Any]],
    override: Optional[int],
) -> int:
    if override is not None:
        return max(1, int(override))
    meta = kernel_db.get("metadata") or {}
    cfg = meta.get("config", meta)
    model_name = cfg.get("model_name") or cfg.get("model")
    if model_name:
        try:
            from transformers import AutoConfig

            hf_cfg = AutoConfig.from_pretrained(model_name)
            nl = getattr(hf_cfg, "num_hidden_layers", None)
            if nl:
                return int(nl)
        except Exception:
            pass
    return _detect_num_layers(classified)


def assert_reference_layer0_layout(
    *,
    layer0_candidates: List[int],
    layer0_decisions: Dict[int, Tuple[bool, str]],
    accepted_chains: List[Tuple[int, int, PairingResult]],
) -> None:
    """Loud checks for the known regression layout (gate at 34 bogus, 44 real MoE block).

    Runs only when layer 0 exposes both execution indices 34 and 44 as moe_gate_proj candidates.
    """
    if 34 not in layer0_candidates or 44 not in layer0_candidates:
        return
    d34 = layer0_decisions.get(34)
    d44 = layer0_decisions.get(44)
    assert d34 is not None and d44 is not None, "internal: missing decisions for reference eis"
    assert d34[0] is False, f"expected reject gate ei=34, got accept ({d34[1]})"
    assert d44[0] is True, f"expected accept gate ei=44, got reject ({d44[1]})"
    pr44 = next((pr for lid, ae, pr in accepted_chains if lid == 0 and ae == 44), None)
    assert pr44 is not None, "expected accepted chain at layer 0 anchor 44"
    assert len(pr44.grouped_mm_eis) == 4, f"expected 4 grouped GEMMs at ei 44 chain, got {pr44.grouped_mm_eis}"
    assert len(pr44.pairs) == 2, f"expected 2 GEMM pairs, got {len(pr44.pairs)}"


def run_minimal_moe_dataflow(
    *,
    classified_kernels: List[Dict[str, Any]],
    output_dir: Path,
    kernel_db_path: Optional[Path] = None,
    trace_path: Optional[Path] = None,
    num_layers: int = 1,
    precision: str = "bfloat16",
    focus_layer: Optional[int] = None,
    moe_debug_log_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Run minimal reconstruction; write ``moe_minimal_*.json`` and ``moe_dataflow.debug.txt``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tp = trace_path
    if tp is None and kernel_db_path is not None:
        tp = resolve_trace_path(Path(kernel_db_path))

    ordered, ordering_source, order_note = order_classified_entries(classified_kernels, tp)
    streams, _ = build_ordered_stream_per_layer(ordered, num_layers=num_layers)

    all_chains: List[Dict[str, Any]] = []
    all_buf_layers: List[Dict[str, Any]] = []
    anchor_lines: List[str] = []
    chain_lines: List[str] = []
    buffer_lines: List[str] = []

    layer0_candidates: List[int] = []
    layer0_decisions: Dict[int, Tuple[bool, str]] = {}
    accepted_for_assert: List[Tuple[int, int, PairingResult]] = []

    for layer_id, layer_ops in enumerate(streams):
        candidates = find_gate_candidates(layer_ops)
        decisions: Dict[int, Tuple[bool, str]] = {}
        for ei in candidates:
            decisions[ei] = validate_gate_anchor(layer_ops, ei)
        if layer_id == 0:
            layer0_candidates = list(candidates)
            layer0_decisions = dict(decisions)

        anchor_lines.extend(
            format_anchor_lines(layer_id, candidates, decisions, focus_layer=focus_layer)
        )
        anchor_lines.append("")

        for ei in candidates:
            ok, reason = decisions[ei]
            if not ok:
                continue
            cw = build_chain_window(layer_ops, ei)
            pr = analyze_chain_window(layer_ops, cw.node_indices, ei)
            accepted_for_assert.append((layer_id, ei, pr))

            chain_record = {
                "layer_id": layer_id,
                "anchor_ei": ei,
                "anchor_validation_forward_ops": ANCHOR_VALIDATION_FORWARD_OPS,
                "window_start_ei": cw.start_ei,
                "window_end_ei": cw.end_ei,
                "pairing": pairing_result_to_json_dict(pr),
            }
            all_chains.append(chain_record)

            cb = build_chain_buffers(layer_ops, ei, pr, precision=precision)
            all_buf_layers.append(
                {
                    "layer_id": layer_id,
                    "anchor_ei": ei,
                    "buffers": cb.buffers,
                    "shape_debug": cb.shape_debug,
                }
            )

            chain_lines.extend(
                format_chain_lines(layer_id, ei, cw, pr, layer_ops, focus_layer=focus_layer)
            )
            chain_lines.append("")
            buffer_lines.extend(format_buffer_lines(cb, focus_layer=focus_layer))
            buffer_lines.append("")

    assert_reference_layer0_layout(
        layer0_candidates=layer0_candidates,
        layer0_decisions=layer0_decisions,
        accepted_chains=accepted_for_assert,
    )

    trace_used = str(tp) if tp and Path(tp).is_file() else None
    dbg = render_debug(
        ordering_source=ordering_source,
        order_note=order_note,
        trace_path_used=trace_used,
        anchor_section=anchor_lines,
        chain_sections=chain_lines,
        buffer_sections=buffer_lines,
    )

    p_chains = output_dir / "moe_minimal_chains.json"
    p_bufs = output_dir / "moe_minimal_buffers.json"
    p_dbg = output_dir / "moe_dataflow.debug.txt"

    p_chains.write_text(
        json.dumps(
            all_chains_to_json(all_chains, ordering_source=ordering_source, order_note=order_note),
            indent=2,
        ),
        encoding="utf-8",
    )
    p_bufs.write_text(json.dumps(all_buffers_to_json(all_buf_layers), indent=2), encoding="utf-8")
    p_dbg.write_text(dbg, encoding="utf-8")

    if moe_debug_log_path is not None:
        from soda.moe.detect import append_moe_op_profile_debug

        append_moe_op_profile_debug(
            moe_debug_log_path,
            f"[soda.moe.moe_dataflow] wrote {p_chains.name}, {p_bufs.name}, {p_dbg.name}",
        )

    return {"moe_minimal_chains": p_chains, "moe_minimal_buffers": p_bufs, "moe_dataflow_debug": p_dbg}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Minimal MoE routed-expert chain reconstruction.")
    parser.add_argument("--kernel-db", type=Path, required=True, help="Path to kernel_database.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for artifacts")
    parser.add_argument("--trace", type=Path, default=None, help="Optional trace.json (overrides sibling of kernel DB)")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="If set, only this layer gets full grouped-GEMM shape detail in debug",
    )
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--moe-num-layers", type=int, default=None)
    args = parser.parse_args()

    kernel_db, kernels = load_kernel_database(args.kernel_db)
    classified = classify_kernels_from_db(kernel_db, kernels)
    num_layers = _resolve_num_layers(kernel_db, classified, args.moe_num_layers)

    trace_path = args.trace
    if trace_path is None:
        trace_path = resolve_trace_path(args.kernel_db)

    paths = run_minimal_moe_dataflow(
        classified_kernels=classified,
        output_dir=args.out_dir,
        kernel_db_path=args.kernel_db,
        trace_path=trace_path,
        num_layers=num_layers,
        precision=args.precision,
        focus_layer=args.layer,
    )
    print("Wrote:", paths)


if __name__ == "__main__":
    _cli()
