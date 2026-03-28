"""Trace-driven MoE execution profile.

`trace.json` (Chrome trace) is the only ordering / layering authority.
Kernel-database classification supplies expert_type / structural_role for NCU
attachment and MoE-specific roles.

Pipeline:
  ordered trace → event typing → launch↔GPU join → parent ATen attach →
  noise filtering → layer segmentation → per-GPU row + NCU attach →
  `execution_trace.json` + aggregated `op_profile.json`.
"""
from __future__ import annotations

import bisect
import json
import math
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from soda.common.data import clean_kernel_name
from soda.moe.debug import debug_print
from soda.moe.detect import append_moe_op_profile_debug, moe_op_profile_debug_path

_GEMM_OPS = frozenset({
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
    "aten::_grouped_mm",
})

_GEMM_STRUCTURAL_ROLES = frozenset({
    "shared_expert_expand",
    "shared_expert_down",
    "routed_expert_expand",
    "routed_expert_down",
    "moe_gate",
    "attention",
    "unknown_gemm",
})

_ROUTED_SUPPORT_STRUCTURAL_ROLES = frozenset({
    "routing_metadata",
    "moe_dispatch_gather",
    "routed_expert_activation",
    "routed_expert_gating_mul",
    "moe_expert_scale",
    "moe_combine_scatter",
    "moe_aux_indexing",
})

_MEANINGFUL_PREFIXES = (
    "aten::rms_norm",
    "aten::native_layer_norm",
    "aten::layer_norm",
    "aten::softmax",
    "aten::_softmax",
    "aten::scaled_dot_product_attention",
    "aten::silu",
    "aten::gelu",
    "aten::relu",
    "aten::mul",
    "aten::add",
    "aten::div",
    "aten::sub",
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
    "aten::_grouped_mm",
    "aten::einsum",
)

_NOISE_OPS = frozenset({
    "aten::empty",
    "aten::empty_strided",
    "aten::empty_like",
    "aten::to",
    "aten::_to_copy",
    "aten::lift_fresh",
    "aten::lift_fresh_copy",
    "aten::as_strided",
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::detach",
    "aten::expand",
    "aten::expand_as",
    "aten::slice",
    "aten::narrow",
    "aten::alias",
    "aten::_reshape_alias",
    "aten::zero_",
    "aten::fill_",
})

_LAYER_ANCHOR_OPS = frozenset({
    "aten::rms_norm",
    "aten::native_layer_norm",
    "aten::layer_norm",
})


# ---------------------------------------------------------------------------
# dtype / HBM helpers (shape estimates when NCU missing)
# ---------------------------------------------------------------------------

def _dtype_bytes(precision: str) -> int:
    _MAP = {
        "bfloat16": 2,
        "float16": 2,
        "fp16": 2,
        "float32": 4,
        "fp32": 4,
        "int8": 1,
        "int4": 1,
    }
    return _MAP.get(precision.lower(), 2)


def _normalize_shape(shape) -> List[int]:
    if shape is None:
        return []
    if isinstance(shape, (int, float)):
        return [int(shape)]
    normalized: List[int] = []
    stack = [shape]
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            for item in reversed(current):
                stack.append(item)
            continue
        if isinstance(current, bool):
            normalized.append(int(current))
            continue
        if isinstance(current, str):
            text = current.strip()
            if text:
                normalized.append(int(float(text)))
            continue
        if isinstance(current, Real):
            normalized.append(int(current))
            continue
        raise TypeError(f"Unsupported shape element type: {type(current)}")
    return normalized


def _product(shape) -> int:
    dims = _normalize_shape(shape)
    if not dims:
        return 0
    r = 1
    for d in dims:
        r *= int(d)
    return r


def _compute_hbm_fields(
    aten_op_name: str,
    input_dims: List,
    dtype_bytes: int,
) -> Dict[str, float]:
    _zero: Dict[str, float] = {
        "flops": 0.0,
        "weight_bytes": 0.0,
        "activation_bytes": 0.0,
        "hbm_bytes": 0.0,
        "kv_bytes": 0.0,
    }
    if not input_dims:
        return _zero

    if aten_op_name == "aten::addmm":
        act_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        weight_shape = _normalize_shape(input_dims[2]) if len(input_dims) > 2 else []
    else:
        act_shape = _normalize_shape(input_dims[0]) if len(input_dims) > 0 else []
        weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []

    if aten_op_name not in _GEMM_OPS or aten_op_name == "aten::_grouped_mm":
        if aten_op_name == "aten::_grouped_mm":
            raw_first = input_dims[0] if input_dims else []
            if (
                raw_first
                and isinstance(raw_first, (list, tuple))
                and raw_first
                and isinstance(raw_first[0], (list, tuple))
            ):
                act_bytes = float(sum(_product(t) for t in raw_first) * dtype_bytes)
            else:
                act_bytes = float(_product(act_shape) * dtype_bytes)
            return {**_zero, "activation_bytes": act_bytes, "hbm_bytes": act_bytes}

        raw_first = input_dims[0] if input_dims else []
        if (
            raw_first
            and isinstance(raw_first, (list, tuple))
            and raw_first
            and isinstance(raw_first[0], (list, tuple))
        ):
            act_bytes = float(sum(_product(t) for t in raw_first) * dtype_bytes)
        else:
            act_bytes = float(_product(act_shape) * dtype_bytes)
        return {**_zero, "activation_bytes": act_bytes, "hbm_bytes": act_bytes}

    if not act_shape or not weight_shape:
        return _zero

    kv_bytes = 0.0

    if aten_op_name == "aten::linear":
        N = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        K = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = float(2 * M * K * N)
        weight_bytes = float(N * K * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name in ("aten::mm", "aten::addmm"):
        K = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        N = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = float(2 * M * K * N)
        weight_bytes = float(K * N * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name == "aten::bmm":
        is_4d = len(act_shape) == 4 or len(weight_shape) == 4
        if is_4d:
            if len(weight_shape) == 4:
                B = int(weight_shape[0])
                H = int(weight_shape[1])
                K = int(weight_shape[2])
                N = int(weight_shape[3])
            else:
                B, H, K, N = 1, 1, 1, 1
            M = int(act_shape[2]) if len(act_shape) > 2 else 1
            flops = float(2 * B * H * M * K * N)
            weight_bytes = 0.0
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * H * M * N)
                * dtype_bytes
            )
            kv_bytes = float(_product(weight_shape) * dtype_bytes)
        else:
            if len(weight_shape) >= 3:
                B = int(weight_shape[0])
                K = int(weight_shape[1])
                N = int(weight_shape[2])
            else:
                B, K, N = 1, 1, 1
            M = int(act_shape[1]) if len(act_shape) > 1 else 1
            flops = float(2 * B * M * K * N)
            weight_bytes = 0.0
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * M * N)
                * dtype_bytes
            )

    elif aten_op_name in ("aten::matmul", "aten::_scaled_mm"):
        if len(act_shape) >= 2 and len(weight_shape) >= 2:
            M = float(_product(act_shape[:-1]))
            K = int(act_shape[-1])
            N = int(weight_shape[-1])
            flops = 2.0 * M * K * N
            weight_bytes = float(_product(weight_shape) * dtype_bytes)
            activation_bytes = float((_product(act_shape) + M * N) * dtype_bytes)
        else:
            return _zero
    else:
        return _zero

    hbm_bytes = weight_bytes + activation_bytes
    return {
        "flops": flops,
        "weight_bytes": weight_bytes,
        "activation_bytes": activation_bytes,
        "hbm_bytes": hbm_bytes,
        "kv_bytes": kv_bytes,
    }


# ---------------------------------------------------------------------------
# Trace parsing & joins
# ---------------------------------------------------------------------------

def _is_meaningful_aten(name: str) -> bool:
    if name in _NOISE_OPS:
        return False
    return any(name.startswith(p) or name == p for p in _MEANINGFUL_PREFIXES)


def _is_trivial_copy(name: str) -> bool:
    if name not in ("aten::copy_", "aten::clone", "aten::contiguous"):
        return False
    return True


def _filtered_meaningful(name: str) -> bool:
    if not _is_meaningful_aten(name):
        return False
    if _is_trivial_copy(name):
        return False
    return True


def _parse_trace_ordered(trace: Dict[str, Any]) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, Dict[str, Any]],
    Dict[int, Dict[str, Any]],
]:
    """Return (aten_ordered, aten_by_ext_id, launches_by_corr)."""
    debug_print("parse_trace_ordered:start", "trace_events=", len(trace.get("traceEvents", [])))
    aten_ordered: List[Dict[str, Any]] = []
    aten_by_ext_id: Dict[int, Dict[str, Any]] = {}
    launches_by_corr: Dict[int, Dict[str, Any]] = {}

    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat = event.get("cat", "")
        name = event.get("name", "")
        args = event.get("args", {}) or {}

        if cat == "cpu_op" and name.startswith("aten::"):
            ext_id = args.get("External id")
            rec = {
                "name": name,
                "ts": event.get("ts", 0),
                "dur": event.get("dur", 0) or 0,
                "external_id": ext_id,
                "input_dims": args.get("Input Dims", []),
                "input_type": args.get("Input type", []),
                "input_strides": args.get("Input Strides", []),
                "concrete_inputs": args.get("Concrete Inputs", []),
                "aten_index": len(aten_ordered),
            }
            aten_ordered.append(rec)
            if ext_id is not None:
                aten_by_ext_id[int(ext_id)] = rec

        elif (cat in ("cuda_runtime", "cuda_driver")) and "LaunchKernel" in name:
            corr = args.get("correlation")
            if corr is not None:
                c = int(corr)
                launches_by_corr[c] = {
                    "name": name,
                    "ts": event.get("ts", 0),
                    "dur": event.get("dur", 0) or 0,
                    "correlation": c,
                    "external_id": args.get("External id"),
                }

        if len(aten_ordered) % 50000 == 0 and len(aten_ordered) > 0:
            debug_print(
                "parse_trace_ordered:progress",
                "aten=", len(aten_ordered),
                "launches=", len(launches_by_corr),
            )

    debug_print(
        "parse_trace_ordered:done",
        "aten=", len(aten_ordered),
        "aten_by_ext=", len(aten_by_ext_id),
        "launches=", len(launches_by_corr),
    )

    return aten_ordered, aten_by_ext_id, launches_by_corr


def _iter_gpu_events_ordered(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GPU kernels and memcpys in strict trace file order."""
    debug_print("iter_gpu_events:start", "trace_events=", len(trace.get("traceEvents", [])))
    out: List[Dict[str, Any]] = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat = event.get("cat", "")
        name = event.get("name", "")
        args = event.get("args", {}) or {}
        ts = event.get("ts", 0)
        dur = event.get("dur", 0) or 0

        if cat == "kernel":
            rec: Dict[str, Any] = {
                "kind": "gpu_kernel",
                "name": name,
                "ts": ts,
                "dur": dur,
                "correlation": args.get("correlation"),
                "external_id": args.get("External id"),
                "stream": args.get("stream"),
                "device": args.get("device"),
                "grid": args.get("grid", [0, 0, 0]),
                "block": args.get("block", [0, 0, 0]),
            }
            for ak, av in args.items():
                if isinstance(ak, str) and ak.startswith("nsys_hbm_"):
                    rec[ak] = av
            out.append(rec)
        elif cat in ("gpu_memcpy", "gpu_memset"):
            mrec: Dict[str, Any] = {
                "kind": "gpu_memcpy",
                "name": name,
                "cat": cat,
                "ts": ts,
                "dur": dur,
                "correlation": args.get("correlation"),
                "stream": args.get("stream"),
                "device": args.get("device"),
            }
            for ak, av in args.items():
                if isinstance(ak, str) and ak.startswith("nsys_hbm_"):
                    mrec[ak] = av
            out.append(mrec)
            if len(out) % 50000 == 0 and len(out) > 0:
                debug_print("iter_gpu_events:progress", "gpu_events=", len(out))
            debug_print("iter_gpu_events:done", "gpu_events=", len(out))
    return out


def _parent_aten(
    gpu_ts: float,
    ext_hint: Optional[int],
    aten_by_ext_id: Dict[int, Dict[str, Any]],
    aten_ordered: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the most likely parent ATen op for a GPU timestamp.

    This function is performance-critical: it can be called millions of times.
    Prefer using the fast path in `build_execution_trace()` which passes in
    precomputed `aten_ts` / `aten_end` arrays.
    """
    if ext_hint is not None:
        rec = aten_by_ext_id.get(int(ext_hint))
        if rec is not None:
            return rec

    # Fallback: build minimal arrays and do a bisect-based lookup.
    # (Still far faster than scanning the entire list for each event.)
    if not aten_ordered:
        return None
    aten_ts = [float(op.get("ts", 0) or 0) for op in aten_ordered]
    aten_end = [float(op.get("ts", 0) or 0) + float(op.get("dur", 0) or 0) for op in aten_ordered]
    return _parent_aten_fast(gpu_ts, aten_ordered, aten_ts, aten_end)


def _parent_aten_fast(
    gpu_ts: float,
    aten_ordered: List[Dict[str, Any]],
    aten_ts: List[float],
    aten_end: List[float],
    *,
    backward_scan_limit: int = 256,
) -> Optional[Dict[str, Any]]:
    """Bisect + bounded backward scan for containing ATen interval.

    Trace events are in file order, and `aten_ordered` is appended in that same
    order, so `aten_ts` is monotonic non-decreasing in practice.
    """
    if not aten_ordered:
        return None

    i = bisect.bisect_right(aten_ts, gpu_ts) - 1
    if i < 0:
        return None

    best_idx = None
    best_dur = None

    # Scan backwards to find the smallest-duration ATen op that *contains* gpu_ts.
    # Overlap depth is typically small; bounding keeps worst-case predictable.
    steps = 0
    j = i
    while j >= 0 and steps < backward_scan_limit:
        if aten_ts[j] <= gpu_ts < aten_end[j]:
            dur = aten_end[j] - aten_ts[j]
            if best_dur is None or dur < best_dur:
                best_dur = dur
                best_idx = j
        # Once the op ends before gpu_ts, earlier ops may still contain it if
        # their duration is larger, so we can't break solely on aten_end[j].
        j -= 1
        steps += 1

    if best_idx is not None:
        return aten_ordered[best_idx]
    # Fallback: closest preceding ATen op.
    return aten_ordered[i]


def _compute_layer_boundaries(
    aten_ordered: List[Dict[str, Any]],
    num_layers: int,
) -> List[int]:
    """Return aten_index where each layer starts (len == num_layers)."""
    debug_print(
        "compute_layer_boundaries:start",
        "aten_count=", len(aten_ordered),
        "num_layers=", num_layers,
    )
    if num_layers <= 0:
        debug_print("compute_layer_boundaries:invalid_num_layers", num_layers)
        return []

    anchor_indices = [
        int(op["aten_index"])
        for op in aten_ordered
        if _filtered_meaningful(op["name"]) and op["name"] in _LAYER_ANCHOR_OPS
    ]

    if len(anchor_indices) >= num_layers and len(anchor_indices) % num_layers == 0:
        step = len(anchor_indices) // num_layers
        out = [anchor_indices[i * step] for i in range(num_layers)]
        debug_print(
            "compute_layer_boundaries:anchor_even",
            "anchors=", len(anchor_indices),
            "step=", step,
            "starts=", out,
        )
        return out

    if len(anchor_indices) >= num_layers:
        step = max(1, len(anchor_indices) // num_layers)
        out = [anchor_indices[min(i * step, len(anchor_indices) - 1)]
               for i in range(num_layers)]
        debug_print(
            "compute_layer_boundaries:anchor_uneven",
            "anchors=", len(anchor_indices),
            "step=", step,
            "starts=", out,
        )
        return out

    meaningful_idx = [
        int(op["aten_index"])
        for op in aten_ordered
        if _filtered_meaningful(op["name"])
    ]
    if len(meaningful_idx) < num_layers:
        out = [0] + [1] * (num_layers - 1)
        debug_print(
            "compute_layer_boundaries:fallback_short_meaningful",
            "meaningful=", len(meaningful_idx),
            "starts=", out,
        )
        return out

    boundaries = []
    chunk = len(meaningful_idx) / num_layers
    for L in range(num_layers):
        slot = int(round(L * chunk))
        slot = min(slot, len(meaningful_idx) - 1)
        boundaries.append(meaningful_idx[slot])
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1
    debug_print(
        "compute_layer_boundaries:fallback_chunked",
        "meaningful=", len(meaningful_idx),
        "starts=", boundaries,
    )
    return boundaries


def _layer_id_for_aten_index(
    aten_index: int,
    layer_starts: List[int],
    num_layers: int,
) -> int:
    if not layer_starts or num_layers <= 0:
        return 0
    pos = bisect.bisect_right(layer_starts, aten_index) - 1
    return int(min(max(pos, 0), num_layers - 1))


def _match_kernel_db_entry(
    cleaned_kernel: str,
    aten_name: str,
    input_dims: List,
    classified: List[Dict],
) -> Optional[Dict]:
    best = None
    best_score = -1
    for entry in classified:
        raw = entry.get("kernel", {}).get("raw_name", "") or ""
        eclean = entry.get("kernel", {}).get("name", "") or ""
        if clean_kernel_name(raw) != cleaned_kernel and eclean != cleaned_kernel:
            continue
        opn = entry.get("aten_op", {}).get("name", "")
        score = 2 if opn == aten_name else 1
        dims = entry.get("aten_op", {}).get("input_dims", [])
        if dims and input_dims and dims == input_dims:
            score += 2
        if score > best_score:
            best_score = score
            best = entry
    return best


def _build_classified_index(classified: List[Dict]) -> Dict[str, List[Dict]]:
    """Index classified entries by cleaned kernel name."""
    debug_print("build_classified_index:start", "classified=", len(classified))
    by_cleaned: Dict[str, List[Dict]] = defaultdict(list)
    for entry in classified:
        raw = entry.get("kernel", {}).get("raw_name", "") or ""
        eclean = entry.get("kernel", {}).get("name", "") or ""
        cleaned = ""
        if eclean:
            cleaned = eclean
        elif raw:
            cleaned = clean_kernel_name(raw)
        if cleaned:
            by_cleaned[cleaned].append(entry)
    debug_print("build_classified_index:done", "unique_cleaned=", len(by_cleaned))
    return dict(by_cleaned)


def _match_kernel_db_entry_indexed(
    cleaned_kernel: str,
    aten_name: str,
    input_dims: List,
    *,
    candidates_by_cleaned: Dict[str, List[Dict]],
) -> Optional[Dict]:
    candidates = candidates_by_cleaned.get(cleaned_kernel)
    if not candidates:
        return None
    best = None
    best_score = -1
    for entry in candidates:
        opn = entry.get("aten_op", {}).get("name", "")
        score = 2 if opn == aten_name else 1
        dims = entry.get("aten_op", {}).get("input_dims", [])
        if dims and input_dims and dims == input_dims:
            score += 2
        if score > best_score:
            best_score = score
            best = entry
    return best


def _infer_semantic_role(
    aten_name: str,
    expert_type: str,
    gemm_structural_role: str,
    input_dims: Optional[List] = None,
) -> str:
    # Legacy semantic role kept for back-compat aggregation/plots.
    if gemm_structural_role == "moe_gate" or expert_type == "gate":
        return "moe_gate"
    if gemm_structural_role == "shared_expert_expand":
        return "shared_expert_expand"
    if gemm_structural_role == "shared_expert_down":
        return "shared_expert_down"
    if gemm_structural_role == "routed_expert_expand":
        return "routed_expert_expand"
    if gemm_structural_role == "routed_expert_down":
        return "routed_expert_down"
    if expert_type == "attention" or gemm_structural_role == "attention":
        return "attention"
    if "softmax" in aten_name or aten_name == "aten::scaled_dot_product_attention":
        return "attention"
    if "norm" in aten_name or "rms_norm" in aten_name:
        return "norm"
    if aten_name in (
        "aten::mul", "aten::add", "aten::div", "aten::sub",
        "aten::silu", "aten::gelu", "aten::relu",
    ):
        return "elementwise"
    if aten_name in _GEMM_OPS:
        if aten_name == "aten::bmm":
            act_shape = _normalize_shape(input_dims[0]) if input_dims else []
            w_shape = _normalize_shape(input_dims[1]) if input_dims and len(input_dims) > 1 else []
            if len(act_shape) == 4 or len(w_shape) == 4:
                return "attention"
            return "bmm"
        if aten_name == "aten::_grouped_mm":
            return "moe_grouped_gemm"
        if expert_type == "routed_expert":
            return "routed_expert_expand"
        if expert_type == "shared_expert":
            return "shared_expert_expand"
        return "linear"
    if aten_name in ("aten::copy_", "aten::clone", "aten::contiguous"):
        return "copy / other"
    return "copy / other"


def _non_gemm_op_family(aten_name: str, cleaned_kernel: str) -> str:
    """Heuristic non-GEMM op-family classifier (broad + debuggable)."""
    a = (aten_name or "").lower()
    k = (cleaned_kernel or "").lower()

    # GPU memcpy/memset pseudo-ops (trace cat) land here with aten_name=="".
    if "memcpy" in k or "memset" in k:
        return "copy_layout"

    if a in ("aten::copy_", "aten::clone", "aten::contiguous", "aten::_to_copy", "aten::to"):
        return "copy_layout"
    if "contiguous" in a or "as_strided" in a or "view" in a or "reshape" in a:
        return "copy_layout"

    if "norm" in a or "rms_norm" in a or a in ("aten::native_layer_norm", "aten::layer_norm"):
        return "normalization"

    if any(x in a for x in ("index_select", "gather", "scatter", "index", "take", "embedding", "lookup")):
        return "indexing_or_routing"

    if any(x in a for x in ("topk", "argsort", "sort", "multinomial")):
        return "routing_metadata"

    if any(x in a for x in ("cat", "concat", "split", "chunk", "stack", "unbind", "pack", "unpack")):
        return "dispatch_combine"

    if any(x in a for x in ("sum", "mean", "amax", "amin", "reduce", "cumsum")):
        return "reduction"

    if a in (
        "aten::mul", "aten::add", "aten::div", "aten::sub",
        "aten::silu", "aten::gelu", "aten::relu", "aten::sigmoid", "aten::tanh",
        "aten::exp", "aten::sqrt",
    ):
        return "elementwise"

    if any(x in a for x in ("softmax", "scaled_dot_product_attention")):
        return "attention_aux"

    return "unknown_non_gemm"


def _initial_structural_role_from_gemm(gemm_structural_role: str) -> str:
    if gemm_structural_role in _GEMM_STRUCTURAL_ROLES:
        return gemm_structural_role
    return "unknown_gemm"


def _initial_structural_role_from_non_gemm_family(family: str) -> str:
    if family == "normalization":
        return "normalization"
    if family == "copy_layout":
        return "copy_layout"
    if family == "indexing_or_routing":
        return "indexing_or_routing"
    if family == "routing_metadata":
        return "routing_metadata"
    if family == "dispatch_combine":
        return "dispatch_combine"
    if family == "attention_aux":
        return "attention_aux"
    if family == "elementwise":
        return "elementwise_misc"
    if family == "reduction":
        return "elementwise_misc"
    return "unknown"


def _byte_class(weight_bytes: float, activation_bytes: float, kv_bytes: float) -> str:
    wb = float(weight_bytes or 0.0)
    ab = float(activation_bytes or 0.0)
    kb = float(kv_bytes or 0.0)
    if kb > 0 and wb == 0 and ab > 0:
        return "kv_cache"
    if wb > 0 and ab > 0:
        return "weight+activation"
    if wb > 0:
        return "weight"
    if ab > 0:
        return "activation"
    return "unknown"


def _placement_class(structural_role: str, byte_class: str) -> str:
    # Keep placement policy orthogonal to structural meaning.
    sr = structural_role or "unknown"
    if sr.startswith("shared_expert_") and byte_class in ("weight", "weight+activation"):
        return "persistent_weight_candidate"
    if sr.startswith("routed_expert_") and byte_class in ("weight", "weight+activation"):
        return "persistent_weight_candidate"
    if sr in ("moe_intermediate", "dispatch_combine", "moe_dispatch_gather", "routed_expert_activation", "routed_expert_gating_mul", "moe_expert_scale", "moe_combine_scatter", "moe_aux_indexing") and byte_class in ("activation", "weight+activation"):
        return "moe_workspace_candidate"
    if sr == "routing_metadata":
        return "routing_metadata"
    if sr in ("attention_aux",) and byte_class == "activation":
        return "attention_workspace_candidate"
    return "non_candidate"


def _sram_candidate_class(placement_class: str) -> str:
    if placement_class in ("persistent_weight_candidate",):
        return "shared_weight"
    if placement_class in ("moe_workspace_candidate",):
        return "moe_workspace"
    if placement_class in ("attention_workspace_candidate",):
        return "attention_workspace"
    return "none"


def _dims_2d(shape: Any) -> Optional[Tuple[int, int]]:
    dims = _normalize_shape(shape)
    if len(dims) != 2:
        return None
    return (int(dims[0]), int(dims[1]))


def _mul_shapes_2d(row: Dict[str, Any]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    dims = row.get("aten_op", {}).get("input_dims", []) or []
    if len(dims) < 2:
        return (None, None)
    return (_dims_2d(dims[0]), _dims_2d(dims[1]))


def _is_gather_like_row(row: Dict[str, Any]) -> bool:
    aten = (row.get("aten_op", {}).get("name") or "").lower()
    kernel = (row.get("gpu_event", {}).get("cleaned_name") or "").lower()
    if aten in ("aten::index", "aten::gather", "aten::index_select", "aten::take"):
        return True
    return "gather" in kernel or "vectorized_gather_kernel" in kernel


def _is_routing_metadata_row(row: Dict[str, Any]) -> bool:
    aten = (row.get("aten_op", {}).get("name") or "").lower()
    kernel = (row.get("gpu_event", {}).get("cleaned_name") or "").lower()
    gpu_name = (row.get("gpu_event", {}).get("name") or "").lower()
    if aten in ("aten::nonzero", "aten::topk", "aten::argsort", "aten::sort"):
        return True
    if "devicecompactinitkernel" in kernel or "deviceselectsweepkernel" in kernel:
        return True
    if "index" in kernel and ("write" in kernel or "put" in kernel):
        return True
    if row.get("kind") == "gpu_memcpy" and "dtoh" in gpu_name and float(row.get("hbm_bytes") or 0.0) <= 65536:
        return True
    return False


def _is_index_add_or_scatter(row: Dict[str, Any]) -> bool:
    aten = (row.get("aten_op", {}).get("name") or "").lower()
    kernel = (row.get("gpu_event", {}).get("cleaned_name") or "").lower()
    if aten == "aten::index_add_" or "scatter" in aten or aten.endswith("scatter_"):
        return True
    return ("index_add" in kernel) or ("scatter" in kernel and "gather" not in kernel)


def _is_other_like_structural_role(role: str) -> bool:
    return role in ("unknown", "copy_layout", "indexing_or_routing", "dispatch_combine", "elementwise_misc")


def _set_row_role(
    row: Dict[str, Any],
    new_role: str,
    source: str,
    confidence: str,
    *,
    allow_override_from: Optional[frozenset] = None,
) -> bool:
    old = row.get("structural_role", "unknown")
    if old == new_role:
        return False
    if allow_override_from is not None and old not in allow_override_from:
        return False
    row["structural_role"] = new_role
    row["classification_source"] = source
    row["classification_confidence"] = confidence
    row["placement_class"] = _placement_class(row["structural_role"], row.get("byte_class", "unknown"))
    row["sram_candidate_class"] = _sram_candidate_class(row["placement_class"])
    return True


def _find_same_layer_window_indices(
    rows: List[Dict[str, Any]],
    i: int,
    *,
    back: int,
    fwd: int,
) -> List[int]:
    lid = rows[i].get("layer_id")
    out: List[int] = []
    lo = max(0, i - back)
    hi = min(len(rows), i + fwd + 1)
    for j in range(lo, hi):
        if rows[j].get("layer_id") == lid:
            out.append(j)
    return out


def _routed_expert_context_relabel(
    rows: List[Dict[str, Any]],
    *,
    window: int = 8,
) -> Dict[str, Any]:
    """Best-effort routed-expert support op relabeling around known routed GEMMs."""
    relabeled: List[Dict[str, Any]] = []
    previous_other_like = {
        int(r.get("exec_index", -1)): r.get("structural_role", "unknown")
        for r in rows
        if _is_other_like_structural_role(r.get("structural_role", "unknown"))
    }

    by_layer: Dict[int, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_layer[int(row.get("layer_id", 0))].append(idx)

    for layer_id, layer_indices in by_layer.items():
        routed_expand = [j for j in layer_indices if rows[j].get("structural_role") == "routed_expert_expand"]
        routed_down = [j for j in layer_indices if rows[j].get("structural_role") == "routed_expert_down"]
        if not routed_expand or not routed_down:
            continue

        for ex_i in routed_expand:
            ex_neighbors = _find_same_layer_window_indices(rows, ex_i, back=window, fwd=window)
            pre_indices = [j for j in ex_neighbors if j < ex_i]
            post_indices = [j for j in ex_neighbors if j > ex_i]

            prev_gather = None
            for j in reversed(pre_indices):
                if _is_gather_like_row(rows[j]):
                    prev_gather = j
                    break
            if prev_gather is not None and float(rows[prev_gather].get("activation_bytes") or 0.0) > 0:
                if _set_row_role(
                    rows[prev_gather],
                    "moe_dispatch_gather",
                    "routed_expert_context",
                    "medium",
                    allow_override_from=frozenset({"unknown", "copy_layout", "indexing_or_routing", "dispatch_combine"}),
                ):
                    relabeled.append({
                        "exec_index": rows[prev_gather].get("exec_index"),
                        "layer_id": layer_id,
                        "old_structural_role": previous_other_like.get(int(rows[prev_gather].get("exec_index", -1)), "other"),
                        "new_structural_role": "moe_dispatch_gather",
                        "aten": rows[prev_gather].get("aten_op", {}).get("name"),
                        "kernel": rows[prev_gather].get("gpu_event", {}).get("cleaned_name"),
                        "hbm_bytes": float(rows[prev_gather].get("hbm_bytes") or 0.0),
                    })

            # Relabel nearby nonzero/compaction metadata on routed path.
            for j in ex_neighbors:
                if j == ex_i:
                    continue
                if _is_routing_metadata_row(rows[j]):
                    if _set_row_role(
                        rows[j],
                        "routing_metadata",
                        "routed_expert_context",
                        "medium",
                        allow_override_from=frozenset({"unknown", "copy_layout", "indexing_or_routing", "dispatch_combine"}),
                    ):
                        relabeled.append({
                            "exec_index": rows[j].get("exec_index"),
                            "layer_id": layer_id,
                            "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                            "new_structural_role": "routing_metadata",
                            "aten": rows[j].get("aten_op", {}).get("name"),
                            "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                            "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                        })

            second_expand = None
            for j in post_indices:
                if rows[j].get("structural_role") == "routed_expert_expand":
                    second_expand = j
                    break
            if second_expand is None:
                continue

            down_idx = None
            for j in post_indices:
                if j <= second_expand:
                    continue
                if rows[j].get("structural_role") == "routed_expert_down":
                    down_idx = j
                    break
            if down_idx is None:
                continue

            # SiLU between expand GEMMs.
            for j in range(ex_i + 1, second_expand):
                if rows[j].get("aten_op", {}).get("name") == "aten::silu":
                    if _set_row_role(
                        rows[j],
                        "routed_expert_activation",
                        "routed_expert_context",
                        "medium",
                        allow_override_from=frozenset({"unknown", "elementwise_misc", "copy_layout"}),
                    ):
                        relabeled.append({
                            "exec_index": rows[j].get("exec_index"),
                            "layer_id": layer_id,
                            "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                            "new_structural_role": "routed_expert_activation",
                            "aten": rows[j].get("aten_op", {}).get("name"),
                            "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                            "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                        })

            # Gating mul between second expand and down.
            for j in range(second_expand + 1, down_idx):
                if rows[j].get("aten_op", {}).get("name") != "aten::mul":
                    continue
                s0, s1 = _mul_shapes_2d(rows[j])
                if s0 is not None and s1 is not None and s0 == s1:
                    if _set_row_role(
                        rows[j],
                        "routed_expert_gating_mul",
                        "routed_expert_context",
                        "medium",
                        allow_override_from=frozenset({"unknown", "elementwise_misc", "copy_layout"}),
                    ):
                        relabeled.append({
                            "exec_index": rows[j].get("exec_index"),
                            "layer_id": layer_id,
                            "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                            "new_structural_role": "routed_expert_gating_mul",
                            "aten": rows[j].get("aten_op", {}).get("name"),
                            "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                            "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                        })

            # Post-down scale + combine/scatter.
            tail_hi = min(len(rows), down_idx + window + 1)
            for j in range(down_idx + 1, tail_hi):
                if rows[j].get("layer_id") != layer_id:
                    continue
                if rows[j].get("aten_op", {}).get("name") == "aten::mul":
                    s0, s1 = _mul_shapes_2d(rows[j])
                    if s0 is not None and s1 is not None:
                        is_scale = (
                            (s0[0] == s1[0] and s0[1] > 1 and s1[1] == 1)
                            or (s1[0] == s0[0] and s1[1] > 1 and s0[1] == 1)
                        )
                        if is_scale and _set_row_role(
                            rows[j],
                            "moe_expert_scale",
                            "routed_expert_context",
                            "medium",
                            allow_override_from=frozenset({"unknown", "elementwise_misc", "copy_layout"}),
                        ):
                            relabeled.append({
                                "exec_index": rows[j].get("exec_index"),
                                "layer_id": layer_id,
                                "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                                "new_structural_role": "moe_expert_scale",
                                "aten": rows[j].get("aten_op", {}).get("name"),
                                "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                                "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                            })

                if _is_index_add_or_scatter(rows[j]):
                    if _set_row_role(
                        rows[j],
                        "moe_combine_scatter",
                        "routed_expert_context",
                        "high",
                        allow_override_from=frozenset({"unknown", "copy_layout", "indexing_or_routing", "dispatch_combine"}),
                    ):
                        relabeled.append({
                            "exec_index": rows[j].get("exec_index"),
                            "layer_id": layer_id,
                            "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                            "new_structural_role": "moe_combine_scatter",
                            "aten": rows[j].get("aten_op", {}).get("name"),
                            "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                            "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                        })
                # Generic aux indexing around routed path.
                elif rows[j].get("aten_op", {}).get("name") in ("aten::index", "aten::gather", "aten::index_select"):
                    if _set_row_role(
                        rows[j],
                        "moe_aux_indexing",
                        "routed_expert_context",
                        "low",
                        allow_override_from=frozenset({"unknown", "copy_layout", "indexing_or_routing"}),
                    ):
                        relabeled.append({
                            "exec_index": rows[j].get("exec_index"),
                            "layer_id": layer_id,
                            "old_structural_role": previous_other_like.get(int(rows[j].get("exec_index", -1)), "other"),
                            "new_structural_role": "moe_aux_indexing",
                            "aten": rows[j].get("aten_op", {}).get("name"),
                            "kernel": rows[j].get("gpu_event", {}).get("cleaned_name"),
                            "hbm_bytes": float(rows[j].get("hbm_bytes") or 0.0),
                        })

    moved_out_bytes = 0.0
    moved_out_count = 0
    for row in rows:
        exec_index = int(row.get("exec_index", -1))
        old = previous_other_like.get(exec_index)
        new = row.get("structural_role", "unknown")
        if old is not None and new in _ROUTED_SUPPORT_STRUCTURAL_ROLES:
            moved_out_bytes += float(row.get("hbm_bytes") or 0.0)
            moved_out_count += 1

    role_bytes = Counter()
    role_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sr = row.get("structural_role", "unknown")
        if sr in _ROUTED_SUPPORT_STRUCTURAL_ROLES:
            role_bytes[sr] += float(row.get("hbm_bytes") or 0.0)
            if len(role_examples[sr]) < 8:
                role_examples[sr].append({
                    "exec_index": row.get("exec_index"),
                    "layer_id": row.get("layer_id"),
                    "aten": row.get("aten_op", {}).get("name"),
                    "kernel": row.get("gpu_event", {}).get("cleaned_name"),
                    "hbm_bytes": float(row.get("hbm_bytes") or 0.0),
                    "classification_source": row.get("classification_source"),
                })

    return {
        "window": window,
        "relabeled_count": len(relabeled),
        "moved_out_of_other_like_count": moved_out_count,
        "moved_out_of_other_like_hbm_bytes": round(moved_out_bytes, 2),
        "hbm_by_new_structural_role": {k: round(v, 2) for k, v in role_bytes.items()},
        "top_examples": dict(role_examples),
        "relabels": relabeled[:400],
    }


def _nearest_confident_roles(
    rows: List[Dict[str, Any]],
    i: int,
    *,
    window: int,
) -> Dict[str, Optional[Tuple[int, str]]]:
    """Find nearest confident structural roles around i (same layer, any stream)."""
    lid = rows[i].get("layer_id")
    prev = None
    nxt = None
    for j in range(i - 1, max(-1, i - window - 1), -1):
        if rows[j].get("layer_id") != lid:
            continue
        sr = rows[j].get("structural_role")
        if sr in _GEMM_STRUCTURAL_ROLES:
            prev = (j, sr)
            break
    for j in range(i + 1, min(len(rows), i + window + 1)):
        if rows[j].get("layer_id") != lid:
            continue
        sr = rows[j].get("structural_role")
        if sr in _GEMM_STRUCTURAL_ROLES:
            nxt = (j, sr)
            break
    return {"prev": prev, "next": nxt}


def _trace_context_relabel(rows: List[Dict[str, Any]], *, window: int = 6) -> List[Dict[str, Any]]:
    """Second pass: relabel non-GEMM ops into trace-meaningful buckets using context.

    Guardrails:
    - Never change confident GEMM structural roles.
    - Only relabel when context evidence is strong.
    """
    relabeled: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        sr = r.get("structural_role", "unknown")
        if sr in _GEMM_STRUCTURAL_ROLES:
            continue
        family = r.get("op_family", "unknown_non_gemm")
        ctx = _nearest_confident_roles(rows, i, window=window)
        prev = ctx["prev"][1] if ctx["prev"] else None
        nxt = ctx["next"][1] if ctx["next"] else None

        new_sr = None

        # Attention neighborhood → attention_aux for nearby non-GEMMs.
        if (prev == "attention" or nxt == "attention") and family in (
            "elementwise", "normalization", "copy_layout", "unknown_non_gemm", "attention_aux",
        ):
            new_sr = "attention_aux"

        # Gate → routed expert expand neighborhood: likely routing/dispatch metadata or glue.
        if new_sr is None and prev == "moe_gate" and nxt in ("routed_expert_expand", "routed_expert_down"):
            if family in ("routing_metadata", "indexing_or_routing", "dispatch_combine"):
                new_sr = "dispatch_combine" if family == "dispatch_combine" else "routing_metadata"
            elif family in ("elementwise", "copy_layout", "unknown_non_gemm", "reduction"):
                new_sr = "routing_metadata"

        # Near routed expert phases: intermediate activations / workspace traffic.
        if new_sr is None and (prev and prev.startswith("routed_expert_") or (nxt and nxt.startswith("routed_expert_"))):
            if family in ("elementwise", "normalization", "copy_layout", "dispatch_combine", "indexing_or_routing", "unknown_non_gemm"):
                new_sr = "moe_intermediate"

        # Near shared expert phases: intermediate activations can still be meaningful.
        if new_sr is None and (prev and prev.startswith("shared_expert_") or (nxt and nxt.startswith("shared_expert_"))):
            if family in ("elementwise", "normalization", "copy_layout", "unknown_non_gemm"):
                new_sr = "moe_intermediate"

        if new_sr is not None and new_sr != sr:
            old = sr
            r["structural_role"] = new_sr
            r["classification_source"] = "trace_context"
            r["classification_confidence"] = "medium"
            relabeled.append({
                "exec_index": r.get("exec_index"),
                "layer_id": r.get("layer_id"),
                "aten": r.get("aten_op", {}).get("name"),
                "kernel": r.get("gpu_event", {}).get("cleaned_name"),
                "op_family": family,
                "old_structural_role": old,
                "new_structural_role": new_sr,
                "prev_confident": prev,
                "next_confident": nxt,
                "hbm_bytes": r.get("hbm_bytes"),
            })
    return relabeled


def detect_num_layers_from_shared_patterns(classified: List[Dict]) -> int:
    """Fallback: GCD of shared-expert invocation counts from kernel DB (not ordering)."""
    debug_print("detect_num_layers:start", "classified=", len(classified))
    shared_freqs = [
        int(e.get("statistics", {}).get("frequency", 0))
        for e in classified
        if e.get("expert_type") == "shared_expert"
        and int(e.get("statistics", {}).get("frequency", 0) or 0) > 0
    ]
    if not shared_freqs:
        debug_print("detect_num_layers:no_shared_freqs")
        return 1
    g = shared_freqs[0]
    for f in shared_freqs[1:]:
        g = math.gcd(g, f)
    out = max(1, g)
    debug_print("detect_num_layers:done", "shared_freqs=", len(shared_freqs), "gcd=", out)
    return out


def build_execution_trace(
    trace_path: Path,
    classified: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Ordered GPU execution rows plus routed-expert relabel debug report."""
    debug_print(
        "build_execution_trace:start",
        "trace_path=", trace_path,
        "classified_count=", len(classified),
        "num_layers=", num_layers,
    )
    trace_path = Path(trace_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        debug_print("build_execution_trace:loading_trace", trace_path)
        trace = json.load(f)
    debug_print("build_execution_trace:trace_loaded")

    aten_ordered, aten_by_ext_id, launches_by_corr = _parse_trace_ordered(trace)
    gpu_events = _iter_gpu_events_ordered(trace)
    dtype_b = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
    candidates_by_cleaned = _build_classified_index(classified)
    aten_ts = [float(op.get("ts", 0) or 0) for op in aten_ordered]
    aten_end = [float(op.get("ts", 0) or 0) + float(op.get("dur", 0) or 0) for op in aten_ordered]
    debug_print(
        "trace_parse:counts",
        "aten_ops=", len(aten_ordered),
        "aten_by_ext=", len(aten_by_ext_id),
        "launches=", len(launches_by_corr),
        "gpu_events=", len(gpu_events),
        "ncu_results=", len(ncu_results),
    )

    layer_starts = _compute_layer_boundaries(aten_ordered, num_layers)
    debug_print("layer_boundaries", layer_starts)

    rows: List[Dict[str, Any]] = [None] * len(gpu_events)  # type: ignore[list-item]
    # Streaming parent-ATen attach: assumes GPU timestamps are usually non-decreasing.
    # Keeps a small active set of ATen ops that overlap the current GPU event.
    aten_i = 0
    active_aten: List[int] = []
    last_gpu_ts: Optional[float] = None
    for exec_index, ev in enumerate(gpu_events):
        if exec_index % 500 == 0:
            debug_print("build_execution_trace:progress", "exec_index=", exec_index, "total=", len(gpu_events))
        corr = ev.get("correlation")
        launch = None
        if corr is not None:
            launch = launches_by_corr.get(int(corr))

        ext = ev.get("external_id")
        if ext is None and launch is not None:
            ext = launch.get("external_id")

        parent = None
        if ext is not None:
            parent = aten_by_ext_id.get(int(ext))
            if parent is not None and exec_index % 10000 == 0:
                debug_print(
                    "build_execution_trace:parent_by_external_id",
                    "exec_index=", exec_index,
                    "external_id=", ext,
                    "aten_index=", parent.get("aten_index"),
                )
        if parent is None:
            gpu_ts = float(ev["ts"])
            # If timestamps go backwards (rare), fall back to bisect method.
            if last_gpu_ts is not None and gpu_ts < last_gpu_ts:
                debug_print(
                    "build_execution_trace:timestamp_regression",
                    "exec_index=", exec_index,
                    "gpu_ts=", gpu_ts,
                    "last_gpu_ts=", last_gpu_ts,
                )
                parent = _parent_aten_fast(gpu_ts, aten_ordered, aten_ts, aten_end)
            else:
                # Advance `aten_i` and add newly-started ops to active list.
                while aten_i < len(aten_ordered) and aten_ts[aten_i] <= gpu_ts:
                    active_aten.append(aten_i)
                    aten_i += 1
                # Drop ops that ended before this gpu_ts.
                if active_aten:
                    active_aten = [idx for idx in active_aten if aten_end[idx] > gpu_ts]
                if exec_index % 10000 == 0:
                    debug_print(
                        "build_execution_trace:active_aten_state",
                        "exec_index=", exec_index,
                        "aten_i=", aten_i,
                        "active_count=", len(active_aten),
                    )
                # Choose the smallest-duration containing op (best parent heuristic).
                best_idx = None
                best_dur = None
                for idx in active_aten:
                    if aten_ts[idx] <= gpu_ts < aten_end[idx]:
                        dur = aten_end[idx] - aten_ts[idx]
                        if best_dur is None or dur < best_dur:
                            best_dur = dur
                            best_idx = idx
                if best_idx is not None:
                    parent = aten_ordered[best_idx]
                elif active_aten:
                    parent = aten_ordered[active_aten[-1]]
                else:
                    parent = None
            last_gpu_ts = gpu_ts
        if parent is None:
            layer_id = 0
            aten_name = ""
            input_dims: List = []
            expert_type = "non_gemm"
            gemm_structural_role = "unknown"
            structural_role = "unknown"
            op_family = "unknown_non_gemm"
            src_entry_id = None
            if exec_index % 10000 == 0:
                debug_print("build_execution_trace:no_parent", "exec_index=", exec_index, "gpu_name=", ev.get("name"))
        else:
            aten_name = parent.get("name", "")
            input_dims = parent.get("input_dims", []) or []
            layer_id = _layer_id_for_aten_index(
                int(parent["aten_index"]), layer_starts, num_layers
            )
            cleaned = clean_kernel_name(ev["name"]) if ev["kind"] == "gpu_kernel" else ""
            match = None
            if ev["kind"] == "gpu_kernel" and cleaned:
                match = _match_kernel_db_entry_indexed(
                    cleaned,
                    aten_name,
                    input_dims,
                    candidates_by_cleaned=candidates_by_cleaned,
                )
            if match is not None:
                expert_type = match.get("expert_type", "non_gemm")
                gemm_structural_role = match.get("gemm_structural_role", "unknown_gemm")
                structural_role = _initial_structural_role_from_gemm(gemm_structural_role)
                op_family = "gemm" if aten_name in _GEMM_OPS else "unknown_non_gemm"
                src_entry_id = match.get("id")
            else:
                expert_type = "non_gemm"
                gemm_structural_role = "non_gemm"
                op_family = _non_gemm_op_family(aten_name, cleaned)
                structural_role = _initial_structural_role_from_non_gemm_family(op_family)
                src_entry_id = None
            if exec_index % 10000 == 0:
                debug_print(
                    "build_execution_trace:parent_attached",
                    "exec_index=", exec_index,
                    "aten=", aten_name,
                    "layer_id=", layer_id,
                    "expert_type=", expert_type,
                    "gemm_structural_role=", gemm_structural_role,
                    "structural_role=", structural_role,
                    "matched_entry=", src_entry_id,
                )

        semantic_role = _infer_semantic_role(
            aten_name, expert_type, gemm_structural_role, input_dims
        )
        if ev["kind"] == "gpu_memcpy":
            semantic_role = "copy / other"
            op_family = "copy_layout"
            gemm_structural_role = "non_gemm"
            structural_role = "copy_layout"

        hbm_fields = _compute_hbm_fields(aten_name, input_dims, dtype_b)
        hbm_byte_data_from_ncu = False
        hbm_bytes = float(hbm_fields["hbm_bytes"])
        if src_entry_id and src_entry_id in ncu_results:
            ncu = ncu_results[src_entry_id]
            ncu_hbm = float(
                (ncu.get("hbm_read_bytes") or 0) + (ncu.get("hbm_write_bytes") or 0)
            )
            if ncu_hbm > 0:
                hbm_bytes = ncu_hbm
                hbm_byte_data_from_ncu = True
                if exec_index % 10000 == 0:
                    debug_print(
                        "build_execution_trace:ncu_override",
                        "exec_index=", exec_index,
                        "src_entry_id=", src_entry_id,
                        "ncu_hbm=", ncu_hbm,
                    )

        grid = ev.get("grid") if ev["kind"] == "gpu_kernel" else None
        cta_count = 1
        if grid:
            for dim in grid:
                cta_count *= int(dim) if dim else 1

        is_shared = expert_type == "shared_expert" if parent is not None else False
        shared_bytes = hbm_bytes if is_shared else 0.0

        row = {
            "exec_index": exec_index,
            "kind": ev["kind"],
            "layer_id": layer_id,
            "semantic_role": semantic_role,
            "op_family": op_family,
            "aten_op": {
                "name": aten_name,
                "input_dims": input_dims,
                "external_id": parent.get("external_id") if parent else None,
                "aten_index": parent.get("aten_index") if parent else None,
            },
            "gpu_event": {
                "name": ev["name"],
                "cleaned_name": cleaned if ev["kind"] == "gpu_kernel" else ev["name"],
                "duration_us": float(ev["dur"]),
                "stream": ev.get("stream"),
                "device": ev.get("device"),
                "correlation": int(corr) if corr is not None else None,
            },
            "cuda_launch": {
                "name": launch.get("name") if launch else None,
                "correlation": launch.get("correlation") if launch else None,
                "duration_us": float(launch["dur"]) if launch else None,
            } if launch else None,
            "flops": float(hbm_fields["flops"]),
            "hbm_bytes": hbm_bytes,
            "HBM_byte_data_from_ncu": hbm_byte_data_from_ncu,
            "weight_bytes": float(hbm_fields["weight_bytes"]),
            "activation_bytes": float(hbm_fields["activation_bytes"]),
            "kv_bytes": float(hbm_fields["kv_bytes"]),
            "shared_expert_bytes": shared_bytes,
            "cta_count": int(cta_count),
            "is_shared_expert": bool(is_shared),
            "expert_type": expert_type,
            "gemm_structural_role": gemm_structural_role,
            "structural_role": structural_role,
            "byte_class": _byte_class(
                float(hbm_fields["weight_bytes"]),
                float(hbm_fields["activation_bytes"]),
                float(hbm_fields["kv_bytes"]),
            ),
            "placement_class": None,  # filled below
            "sram_candidate_class": None,  # filled below
            "classification_source": "gemm_shape" if structural_role in _GEMM_STRUCTURAL_ROLES else "non_gemm_heuristic",
            "classification_confidence": "high" if structural_role in _GEMM_STRUCTURAL_ROLES else "low",
            "source_kernel_db_entry_id": src_entry_id,
        }
        row["placement_class"] = _placement_class(row["structural_role"], row["byte_class"])
        row["sram_candidate_class"] = _sram_candidate_class(row["placement_class"])
        for _k, _v in ev.items():
            if isinstance(_k, str) and _k.startswith("nsys_hbm_"):
                row[_k] = _v
        rows[exec_index] = row
    debug_print(
        "build_execution_trace:summary",
        "rows=", len(rows),
        "kernels=", sum(1 for r in rows if r and r.get("kind") == "gpu_kernel"),
        "memcpy=", sum(1 for r in rows if r and r.get("kind") == "gpu_memcpy"),
    )
    relabeled = _routed_expert_context_relabel(rows, window=8)
    debug_print(
        "build_execution_trace:context_relabel",
        "relabeled_count=", relabeled.get("relabeled_count", 0),
    )
    debug_print("build_execution_trace:done", "rows=", len(rows))
    return rows, relabeled


def _traffic_summaries(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Debug/inspection payload: HBM by structural/byte/placement + unknown drilldown."""
    by_structural = Counter()
    by_byte_class = Counter()
    by_placement = Counter()
    unknown_aten = Counter()
    unknown_kernel = Counter()

    for r in rows:
        hb = float(r.get("hbm_bytes") or 0.0)
        by_structural[r.get("structural_role", "unknown")] += hb
        by_byte_class[r.get("byte_class", "unknown")] += hb
        by_placement[r.get("placement_class", "unknown")] += hb

        if r.get("structural_role") == "unknown":
            unknown_aten[r.get("aten_op", {}).get("name") or ""] += hb
            unknown_kernel[r.get("gpu_event", {}).get("cleaned_name") or ""] += hb

    def _top(counter: Counter, n: int = 20) -> List[Dict[str, Any]]:
        return [{"name": k, "hbm_bytes": round(v, 2)} for k, v in counter.most_common(n)]

    return {
        "hbm_by_structural_role": {k: round(v, 2) for k, v in by_structural.items()},
        "hbm_by_byte_class": {k: round(v, 2) for k, v in by_byte_class.items()},
        "hbm_by_placement_class": {k: round(v, 2) for k, v in by_placement.items()},
        "unknown_top_aten_ops": _top(unknown_aten, 30),
        "unknown_top_cleaned_kernels": _top(unknown_kernel, 30),
    }


def _aggregate(rows: List[Dict[str, Any]], num_layers: int) -> Dict[str, Any]:
    debug_print("aggregate:start", "rows=", len(rows), "num_layers=", num_layers)
    per_layer: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "kernel_rows": 0,
        "memcpy_rows": 0,
        "time_us": 0.0,
        "hbm_bytes_ncu": 0.0,
        "hbm_bytes_estimated": 0.0,
        "by_role": Counter(),
        "by_structural_role": Counter(),
        "by_placement_class": Counter(),
    })
    per_role: Counter = Counter()
    per_structural: Counter = Counter()
    per_placement: Counter = Counter()
    expert_hbm = defaultdict(float)
    expert_time = defaultdict(float)

    total_ncu = 0.0
    total_est = 0.0

    for r in rows:
        lid = int(r["layer_id"])
        pl = per_layer[lid]
        if r["kind"] == "gpu_kernel":
            pl["kernel_rows"] += 1
        else:
            pl["memcpy_rows"] += 1
        du = float(r["gpu_event"]["duration_us"])
        pl["time_us"] += du
        hb = float(r["hbm_bytes"])
        if r.get("HBM_byte_data_from_ncu"):
            pl["hbm_bytes_ncu"] += hb
            total_ncu += hb
        else:
            pl["hbm_bytes_estimated"] += hb
            total_est += hb
        role = r.get("semantic_role", "other")
        pl["by_role"][role] += 1
        per_role[role] += hb
        sr = r.get("structural_role", "unknown")
        pl["by_structural_role"][sr] += hb
        per_structural[sr] += hb
        pc = r.get("placement_class", "unknown")
        pl["by_placement_class"][pc] += hb
        per_placement[pc] += hb
        et = r.get("expert_type", "other")
        if et in ("shared_expert", "routed_expert", "gate"):
            expert_hbm[et] += hb
            expert_time[et] += du
        if (pl["kernel_rows"] + pl["memcpy_rows"]) % 50000 == 0 and (pl["kernel_rows"] + pl["memcpy_rows"]) > 0:
            debug_print(
                "aggregate:layer_progress",
                "layer=", lid,
                "rows=", pl["kernel_rows"] + pl["memcpy_rows"],
                "time_us=", pl["time_us"],
            )

    layer_list = []
    for L in range(num_layers):
        d = dict(per_layer[L])
        d["layer_id"] = L
        d["by_role"] = dict(d["by_role"])
        d["by_structural_role"] = dict(d["by_structural_role"])
        d["by_placement_class"] = dict(d["by_placement_class"])
        layer_list.append(d)

    return {
        "per_layer": layer_list,
        "per_semantic_role": dict(per_role),
        "per_structural_role": {k: round(v, 2) for k, v in per_structural.items()},
        "per_placement_class": {k: round(v, 2) for k, v in per_placement.items()},
        "expert_totals": {
            k: {"hbm_bytes": round(v, 2), "time_us": round(expert_time[k], 2)}
            for k, v in expert_hbm.items()
        },
        "hbm_breakdown": {
            "from_ncu_sum": round(total_ncu, 2),
            "estimated_sum": round(total_est, 2),
            "combined_sum": round(total_ncu + total_est, 2),
        },
    }


def generate_op_profile(
    trace_path: Path,
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path] = None,
    execution_trace_path: Optional[Path] = None,
    moe_debug_log_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build execution_trace.json + aggregated op_profile.json from trace.json."""
    debug_print(
        "op_profile_gen:start",
        "trace_path=", trace_path,
        "num_layers=", num_layers,
        "precision=", precision,
        "classified_count=", len(classified_kernels),
        "ncu_count=", len(ncu_results or {}),
    )
    if output_path is not None and moe_debug_log_path is None:
        moe_debug_log_path = moe_op_profile_debug_path(output_path)

    trace_path = Path(trace_path)
    num_layers = max(1, int(num_layers))

    append_moe_op_profile_debug(
        moe_debug_log_path,
        f"[moe.op_profile] trace-centric build trace={trace_path} num_layers={num_layers}",
    )

    rows, routed_relabel_report = build_execution_trace(
        trace_path=trace_path,
        classified=classified_kernels,
        num_layers=num_layers,
        precision=precision,
        ncu_results=ncu_results,
    )
    debug_print("op_profile_gen:rows_built", len(rows))

    if execution_trace_path is None and output_path is not None:
        execution_trace_path = Path(output_path).parent / "execution_trace.json"

    if execution_trace_path is not None:
        et_path = Path(execution_trace_path)
        et_path.parent.mkdir(parents=True, exist_ok=True)
        et_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        debug_print("op_profile_gen:execution_trace_written", et_path)

    # Always emit a debug summary alongside op_profile.json when possible.
    summaries = _traffic_summaries(rows)
    if output_path is not None:
        summary_path = Path(output_path).with_name("trace_label_summary.json")
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        debug_print("op_profile_gen:trace_label_summary_written", summary_path)
        append_moe_op_profile_debug(
            moe_debug_log_path,
            f"[moe.op_profile] wrote trace label summary {summary_path}",
        )

        # Emit routed-expert relabel debug report for support-op verification.
        try:
            relabel_path = Path(output_path).with_name("routed_expert_relabel_report.json")
            relabel_path.write_text(json.dumps(routed_relabel_report, indent=2), encoding="utf-8")
            debug_print("op_profile_gen:routed_relabel_report_written", relabel_path)
            append_moe_op_profile_debug(
                moe_debug_log_path,
                f"[moe.op_profile] wrote routed-expert relabel report {relabel_path}",
            )
        except Exception as e:
            debug_print("op_profile_gen:routed_relabel_report_write_failed", str(e))

    aggregates = _aggregate(rows, num_layers)
    debug_print("op_profile_gen:aggregated", "keys=", list(aggregates.keys()))

    profile = {
        "schema_version": 2,
        "source_trace": str(trace_path.resolve()),
        "execution_trace_file": str(Path(execution_trace_path).resolve())
        if execution_trace_path else None,
        "num_layers": num_layers,
        "row_count": len(rows),
        "aggregates": aggregates,
        "trace_label_summary_file": str(Path(output_path).with_name("trace_label_summary.json").resolve())
        if output_path is not None else None,
        "routed_expert_relabel_report_file": str(Path(output_path).with_name("routed_expert_relabel_report.json").resolve())
        if output_path is not None else None,
    }

    nsys_agg = {
        "nsys_hbm_estimated_read_bytes": 0.0,
        "nsys_hbm_estimated_write_bytes": 0.0,
        "nsys_hbm_estimated_total_bytes": 0.0,
    }
    for r in rows:
        for nk in nsys_agg:
            nv = r.get(nk)
            if nv is not None:
                nsys_agg[nk] += float(nv)
    if any(v > 0 for v in nsys_agg.values()):
        profile["nsys_hbm_aggregates"] = {k: round(v, 2) for k, v in nsys_agg.items()}
        profile["nsys_hbm_estimated_total_gb"] = round(
            nsys_agg["nsys_hbm_estimated_total_bytes"] / 1e9, 6
        )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(profile, indent=2),
            encoding="utf-8",
        )
        debug_print("op_profile_gen:profile_written", output_path)

    debug_print("op_profile_gen:done", "row_count=", len(rows))
    return profile


# Back-compat alias for pipeline tests / old imports
