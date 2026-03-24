"""CLI and orchestration for minimal MoE routed-expert chain reconstruction.

This path does not call ``soda.moe.dataflow`` (legacy broad grouping). Usage::

    python -m soda.moe.moe_dataflow.main --kernel-db ... --out-dir ... [--trace ...] [--layer N] [--debug-full-layer]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    format_section_a_candidate_gates,
    format_section_b_routed_windows,
    format_section_c_gemm_pairs,
    format_section_d_buffers,
    render_debug,
)
from soda.moe.moe_dataflow.loader import classify_kernels_from_db, load_kernel_database
from soda.moe.moe_dataflow.ordering import build_ordered_stream_per_layer, order_classified_entries
from soda.moe.moe_dataflow.pairing import PairingResult, analyze_chain_window
from soda.moe.moe_dataflow.windows import ChainWindow, build_chain_window
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
    """Regression layout: gate 34 bogus, 44 real; four grouped GEMMs; two pairs."""
    if 34 not in layer0_candidates or 44 not in layer0_candidates:
        return
    d34 = layer0_decisions.get(34)
    d44 = layer0_decisions.get(44)
    assert d34 is not None and d44 is not None, "internal: missing decisions for reference eis"
    assert d34[0] is False, f"expected reject gate ei=34, got accept ({d34[1]})"
    assert d44[0] is True, f"expected accept gate ei=44, got reject ({d44[1]})"
    pr44 = next((pr for lid, ae, pr in accepted_chains if lid == 0 and ae == 44), None)
    assert pr44 is not None, "expected accepted chain at layer 0 anchor 44"
    assert pr44.grouped_mm_eis == [50, 51, 52, 53], (
        f"expected grouped GEMMs [50,51,52,53], got {pr44.grouped_mm_eis}"
    )
    assert len(pr44.pairs) == 2, f"expected 2 GEMM pairs, got {len(pr44.pairs)}"


def _flat_chain_record(
    layer_id: int,
    ei: int,
    cw: ChainWindow,
    pr: PairingResult,
) -> Dict[str, Any]:
    sel = [c.execution_index for c in pr.classified if c.coarse_class == "routing_select"]
    meta = [c.execution_index for c in pr.classified if c.coarse_class == "routing_metadata"]
    return {
        "layer_id": layer_id,
        "anchor_ei": ei,
        "anchor_validation_forward_ops": ANCHOR_VALIDATION_FORWARD_OPS,
        "window_start_ei": cw.start_ei,
        "window_end_ei": cw.end_ei,
        "routing_select_eis": sel,
        "routing_metadata_eis": meta,
        "grouped_gemm_eis": list(pr.grouped_mm_eis),
        "gemm_pairs": [
            {"pair_id": p.pair_id, "gemm0_ei": p.gemm0_ei, "gemm1_ei": p.gemm1_ei}
            for p in pr.pairs
        ],
        "odd_unpaired_grouped_mm_dropped": pr.odd_gemm_warning,
    }


def run_minimal_moe_dataflow(
    *,
    classified_kernels: List[Dict[str, Any]],
    output_dir: Path,
    kernel_db_path: Optional[Path] = None,
    trace_path: Optional[Path] = None,
    num_layers: int = 1,
    precision: str = "bfloat16",
    focus_layer: Optional[int] = None,
    debug_full_layer: bool = False,
    moe_debug_log_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Minimal reconstruction; writes ``moe_minimal_*.json`` and ``moe_dataflow.debug.txt``.

    If ``focus_layer`` is set, only that layer is parsed and emitted (iteration mode).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tp = trace_path
    if tp is None and kernel_db_path is not None:
        tp = resolve_trace_path(Path(kernel_db_path))

    ordered, ordering_source, order_note = order_classified_entries(classified_kernels, tp)
    streams, _ = build_ordered_stream_per_layer(ordered, num_layers=num_layers)

    single_layer_mode = focus_layer is not None
    layer_indices: List[int]
    if single_layer_mode:
        lid = int(focus_layer)
        if lid < 0 or lid >= len(streams):
            print(
                f"[moe_dataflow] WARNING: --layer {lid} out of range (0..{len(streams)-1}); no output",
                file=sys.stderr,
            )
            layer_indices = []
        else:
            layer_indices = [lid]
    else:
        layer_indices = list(range(len(streams)))

    all_chains: List[Dict[str, Any]] = []
    all_buf_layers: List[Dict[str, Any]] = []
    section_a: List[str] = []
    section_b: List[str] = []
    section_c: List[str] = []
    section_d: List[str] = []

    layer0_candidates: List[int] = []
    layer0_decisions: Dict[int, Tuple[bool, str]] = {}
    accepted_for_assert: List[Tuple[int, int, PairingResult]] = []

    for layer_id in layer_indices:
        layer_ops = streams[layer_id]
        candidates = find_gate_candidates(layer_ops)
        decisions: Dict[int, Tuple[bool, str]] = {}
        for ei in candidates:
            decisions[ei] = validate_gate_anchor(layer_ops, ei)
        if layer_id == 0:
            layer0_candidates = list(candidates)
            layer0_decisions = dict(decisions)

        section_a.extend(
            format_section_a_candidate_gates(
                layer_id, candidates, decisions, single_layer_mode=single_layer_mode
            )
        )

        for ei in candidates:
            ok, reason = decisions[ei]
            if not ok:
                continue
            cw = build_chain_window(layer_ops, ei)
            pr = analyze_chain_window(layer_ops, cw.node_indices, ei)
            accepted_for_assert.append((layer_id, ei, pr))

            all_chains.append(_flat_chain_record(layer_id, ei, cw, pr))

            cb = build_chain_buffers(layer_ops, ei, pr, precision=precision)
            all_buf_layers.append(
                {
                    "layer_id": layer_id,
                    "anchor_ei": ei,
                    "buffers": cb.buffers,
                    "shape_debug": cb.shape_debug,
                }
            )

            section_b.extend(
                format_section_b_routed_windows(
                    layer_id,
                    ei,
                    cw,
                    pr,
                    layer_ops,
                    debug_full_layer=debug_full_layer,
                )
            )
            section_b.append("")
            section_c.extend(
                format_section_c_gemm_pairs(
                    layer_id,
                    ei,
                    pr,
                    layer_ops,
                    debug_full_layer=debug_full_layer,
                )
            )
            section_c.append("")
            section_d.extend(format_section_d_buffers(cb))

    if not single_layer_mode or focus_layer == 0:
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
        section_a=section_a,
        section_b=section_b,
        section_c=section_c,
        section_d=section_d,
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
        help="Parse and emit only this layer (compact debug for that layer)",
    )
    parser.add_argument(
        "--debug-full-layer",
        action="store_true",
        help="Include per-grouped-GEMM shape lines in Sections B/C",
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
        debug_full_layer=args.debug_full_layer,
    )
    print("Wrote:", paths)


if __name__ == "__main__":
    _cli()
