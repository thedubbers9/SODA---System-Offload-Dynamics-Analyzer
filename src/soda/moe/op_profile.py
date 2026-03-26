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

    return aten_ordered, aten_by_ext_id, launches_by_corr


def _iter_gpu_events_ordered(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """GPU kernels and memcpys in strict trace file order."""
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
            out.append({
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
            })
        elif cat in ("gpu_memcpy", "gpu_memset"):
            out.append({
                "kind": "gpu_memcpy",
                "name": name,
                "cat": cat,
                "ts": ts,
                "dur": dur,
                "correlation": args.get("correlation"),
                "stream": args.get("stream"),
                "device": args.get("device"),
            })
    return out


def _parent_aten(
    gpu_ts: float,
    ext_hint: Optional[int],
    aten_by_ext_id: Dict[int, Dict[str, Any]],
    aten_ordered: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if ext_hint is not None and int(ext_hint) in aten_by_ext_id:
        return aten_by_ext_id[int(ext_hint)]

    candidates = [
        op for op in aten_ordered
        if op["ts"] <= gpu_ts < op["ts"] + op["dur"]
    ]
    if candidates:
        return min(candidates, key=lambda o: o["dur"])
    prev = None
    for op in aten_ordered:
        if op["ts"] <= gpu_ts:
            prev = op
    return prev


def _compute_layer_boundaries(
    aten_ordered: List[Dict[str, Any]],
    num_layers: int,
) -> List[int]:
    """Return aten_index where each layer starts (len == num_layers)."""
    if num_layers <= 0:
        return []

    anchor_indices = [
        int(op["aten_index"])
        for op in aten_ordered
        if _filtered_meaningful(op["name"]) and op["name"] in _LAYER_ANCHOR_OPS
    ]

    if len(anchor_indices) >= num_layers and len(anchor_indices) % num_layers == 0:
        step = len(anchor_indices) // num_layers
        return [anchor_indices[i * step] for i in range(num_layers)]

    if len(anchor_indices) >= num_layers:
        step = max(1, len(anchor_indices) // num_layers)
        return [anchor_indices[min(i * step, len(anchor_indices) - 1)]
                for i in range(num_layers)]

    meaningful_idx = [
        int(op["aten_index"])
        for op in aten_ordered
        if _filtered_meaningful(op["name"])
    ]
    if len(meaningful_idx) < num_layers:
        return [0] + [1] * (num_layers - 1)

    boundaries = []
    chunk = len(meaningful_idx) / num_layers
    for L in range(num_layers):
        slot = int(round(L * chunk))
        slot = min(slot, len(meaningful_idx) - 1)
        boundaries.append(meaningful_idx[slot])
    for i in range(1, len(boundaries)):
        if boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + 1
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


def _infer_semantic_role(
    aten_name: str,
    expert_type: str,
    structural_role: str,
    input_dims: Optional[List] = None,
) -> str:
    if structural_role == "moe_gate" or expert_type == "gate":
        return "moe_gate"
    if structural_role == "shared_expert_expand":
        return "shared_expert_expand"
    if structural_role == "shared_expert_down":
        return "shared_expert_down"
    if structural_role == "routed_expert_expand":
        return "routed_expert_expand"
    if structural_role == "routed_expert_down":
        return "routed_expert_down"
    if expert_type == "attention" or structural_role == "attention":
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


def detect_num_layers_from_shared_patterns(classified: List[Dict]) -> int:
    """Fallback: GCD of shared-expert invocation counts from kernel DB (not ordering)."""
    shared_freqs = [
        int(e.get("statistics", {}).get("frequency", 0))
        for e in classified
        if e.get("expert_type") == "shared_expert"
        and int(e.get("statistics", {}).get("frequency", 0) or 0) > 0
    ]
    if not shared_freqs:
        return 1
    g = shared_freqs[0]
    for f in shared_freqs[1:]:
        g = math.gcd(g, f)
    return max(1, g)


def build_execution_trace(
    trace_path: Path,
    classified: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """Ordered GPU execution rows with layer_id, semantic role, optional NCU."""
    debug_print(
        "build_execution_trace:start",
        "trace_path=", trace_path,
        "classified_count=", len(classified),
        "num_layers=", num_layers,
    )
    trace_path = Path(trace_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    aten_ordered, aten_by_ext_id, launches_by_corr = _parse_trace_ordered(trace)
    gpu_events = _iter_gpu_events_ordered(trace)
    dtype_b = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
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

    rows: List[Dict[str, Any]] = []
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

        parent = _parent_aten(ev["ts"], ext, aten_by_ext_id, aten_ordered)
        if parent is None:
            layer_id = 0
            aten_name = ""
            input_dims: List = []
            expert_type = "other"
            structural_role = "other"
            src_entry_id = None
        else:
            aten_name = parent.get("name", "")
            input_dims = parent.get("input_dims", []) or []
            layer_id = _layer_id_for_aten_index(
                int(parent["aten_index"]), layer_starts, num_layers
            )
            cleaned = clean_kernel_name(ev["name"]) if ev["kind"] == "gpu_kernel" else ""
            match = None
            if ev["kind"] == "gpu_kernel" and cleaned:
                match = _match_kernel_db_entry(cleaned, aten_name, input_dims, classified)
            if match is not None:
                expert_type = match.get("expert_type", "other")
                structural_role = match.get("structural_role", "other")
                src_entry_id = match.get("id")
            else:
                expert_type = "other"
                structural_role = "other"
                src_entry_id = None

        semantic_role = _infer_semantic_role(
            aten_name, expert_type, structural_role, input_dims
        )
        if ev["kind"] == "gpu_memcpy":
            semantic_role = "copy / other"

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
            "aten_op": {
                "name": aten_name,
                "input_dims": input_dims,
                "external_id": parent.get("external_id") if parent else None,
                "aten_index": parent.get("aten_index") if parent else None,
            },
            "gpu_event": {
                "name": ev["name"],
                "cleaned_name": clean_kernel_name(ev["name"])
                if ev["kind"] == "gpu_kernel"
                else ev["name"],
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
            "structural_role": structural_role,
            "source_kernel_db_entry_id": src_entry_id,
        }
        rows.append(row)
    debug_print("build_execution_trace:done", "rows=", len(rows))
    return rows


def _aggregate(rows: List[Dict[str, Any]], num_layers: int) -> Dict[str, Any]:
    debug_print("aggregate:start", "rows=", len(rows), "num_layers=", num_layers)
    per_layer: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "kernel_rows": 0,
        "memcpy_rows": 0,
        "time_us": 0.0,
        "hbm_bytes_ncu": 0.0,
        "hbm_bytes_estimated": 0.0,
        "by_role": Counter(),
    })
    per_role: Counter = Counter()
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
        et = r.get("expert_type", "other")
        if et in ("shared_expert", "routed_expert", "gate"):
            expert_hbm[et] += hb
            expert_time[et] += du

    layer_list = []
    for L in range(num_layers):
        d = dict(per_layer[L])
        d["layer_id"] = L
        d["by_role"] = dict(d["by_role"])
        layer_list.append(d)

    return {
        "per_layer": layer_list,
        "per_semantic_role": dict(per_role),
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

    rows = build_execution_trace(
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
    }

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
