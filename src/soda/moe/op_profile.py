"""Per-layer per-op kernel profile generator.

Generates op_profile.json: one record per unique kernel invocation type per layer
across the full model, with is_shared_expert flag for downstream filtering.

Data source: classified kernel DB entries (from classify_kernel_entries()).
No GPU required — derived from kernel_database.json + MoE classification.
"""
from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ATen ops that represent GEMM-type operations (matches detect.py GEMM_OPS).
_GEMM_OPS = frozenset({
    "aten::linear",
    "aten::mm",
    "aten::bmm",
    "aten::addmm",
    "aten::matmul",
    "aten::_scaled_mm",
})


# ---------------------------------------------------------------------------
# Public helpers (exported for tests)
# ---------------------------------------------------------------------------

def _dtype_bytes(precision: str) -> int:
    """Return bytes per element for the given precision string."""
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


def _compute_hbm_fields(
    aten_op_name: str,
    input_dims: List,
    dtype_bytes: int,
) -> Dict:
    """Compute weight_bytes, activation_bytes, hbm_bytes, kv_bytes, flops.

    For GEMM ops, derives M, K, N from input shapes and returns shape-based
    estimates.  For non-GEMM, returns activation bytes from first input only.

    Args:
        aten_op_name: ATen op name (e.g., "aten::linear").
        input_dims:   input_dims list from kernel DB entry's aten_op dict.
        dtype_bytes:  Bytes per element for the run precision.

    Returns:
        Dict with keys: flops, weight_bytes, activation_bytes, hbm_bytes, kv_bytes.
    """
    _zero = {"flops": 0, "weight_bytes": 0.0, "activation_bytes": 0.0,
             "hbm_bytes": 0.0, "kv_bytes": 0.0}

    if not input_dims:
        return _zero

    # addmm(bias, input, weight, ...) → activation at [1], weight at [2]
    if aten_op_name == "aten::addmm":
        act_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        weight_shape = _normalize_shape(input_dims[2]) if len(input_dims) > 2 else []
    else:
        act_shape = _normalize_shape(input_dims[0]) if len(input_dims) > 0 else []
        weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []

    if aten_op_name not in _GEMM_OPS:
        # Non-GEMM: report activation bytes from first input.
        # Some ops (aten::cat, aten::stack) receive a *list of tensors* as
        # input_dims[0], e.g. [[1,16,1024,64],[1,16,1024,64]].  Detect this
        # (list-of-lists) and sum each tensor's bytes individually instead of
        # blindly flattening into one huge product.
        raw_first = input_dims[0] if input_dims else []
        if raw_first and isinstance(raw_first, (list, tuple)) and raw_first and isinstance(raw_first[0], (list, tuple)):
            # List of tensor shapes: sum each tensor's bytes.
            act_bytes = float(sum(_product(t) for t in raw_first) * dtype_bytes)
        else:
            act_bytes = float(_product(act_shape) * dtype_bytes)
        return {**_zero, "activation_bytes": act_bytes, "hbm_bytes": act_bytes}

    if not act_shape or not weight_shape:
        return _zero

    kv_bytes = 0.0

    if aten_op_name == "aten::linear":
        # weight stored as (N, K); input as (..., K).
        N = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        K = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        # Flatten all batch dims of the activation.
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = 2 * M * K * N
        weight_bytes = float(N * K * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name in ("aten::mm", "aten::addmm"):
        # mm:    (M, K) × (K, N)  → weight_shape = (K, N)
        # addmm: bias, (M, K), (K, N) → weight_shape = (K, N) (extracted above)
        K = int(weight_shape[0]) if len(weight_shape) > 0 else 1
        N = int(weight_shape[1]) if len(weight_shape) > 1 else 1
        M = _product(act_shape[:-1]) if len(act_shape) > 1 else int(act_shape[0])
        flops = 2 * M * K * N
        weight_bytes = float(K * N * dtype_bytes)
        activation_bytes = float((M * K + M * N) * dtype_bytes)

    elif aten_op_name == "aten::bmm":
        # Standard 3D: (B, M, K) × (B, K, N)
        # Attention 4D: (B, H, M, K) × (B, H, K, N)
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
            flops = 2 * B * H * M * K * N
            weight_bytes = 0.0  # both tensors are activations in attention
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * H * M * N)
                * dtype_bytes
            )
            kv_bytes = float(_product(weight_shape) * dtype_bytes)
        else:
            # 3D bmm
            if len(weight_shape) >= 3:
                B = int(weight_shape[0])
                K = int(weight_shape[1])
                N = int(weight_shape[2])
            else:
                B, K, N = 1, 1, 1
            M = int(act_shape[1]) if len(act_shape) > 1 else 1
            flops = 2 * B * M * K * N
            weight_bytes = 0.0
            activation_bytes = float(
                (_product(act_shape) + _product(weight_shape) + B * M * N)
                * dtype_bytes
            )

    elif aten_op_name in ("aten::matmul", "aten::_scaled_mm"):
        if len(act_shape) >= 2 and len(weight_shape) >= 2:
            M = _product(act_shape[:-1])
            K = int(act_shape[-1])
            N = int(weight_shape[-1])
            flops = 2 * M * K * N
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


def _infer_structural_op_name(
    aten_op_name: str,
    structural_role: str,
    input_dims: List,
) -> str:
    """Map (aten_op, structural_role, input_dims) to a human-readable op_name."""
    # Non-GEMM special cases.
    if aten_op_name in ("aten::rms_norm",):
        return "rmsnorm"
    if aten_op_name in ("aten::native_layer_norm", "aten::layer_norm"):
        return "layernorm"
    if aten_op_name in ("aten::softmax", "aten::_softmax"):
        return "softmax"
    if aten_op_name in ("aten::silu", "aten::gelu", "aten::relu"):
        return "activation"
    if aten_op_name in ("aten::mul", "aten::add", "aten::div", "aten::sub"):
        return "elementwise"

    # bmm: check 4D vs 3D.
    if aten_op_name == "aten::bmm":
        act_shape = _normalize_shape(input_dims[0]) if input_dims else []
        weight_shape = _normalize_shape(input_dims[1]) if len(input_dims) > 1 else []
        if len(act_shape) == 4 or len(weight_shape) == 4:
            return "attn_bmm_kv"
        return "bmm"

    # GEMM ops: classify by structural role.
    if aten_op_name in _GEMM_OPS:
        if structural_role == "shared_expert_expand":
            return "shared_expert_expand"
        if structural_role == "shared_expert_down":
            return "shared_expert_down"
        if structural_role == "routed_expert_expand":
            return "routed_expert_expand"
        if structural_role == "routed_expert_down":
            return "routed_expert_down"
        if structural_role == "moe_gate":
            return "moe_gate_proj"
        if structural_role == "attention":
            return "attn_proj"
        return "linear"

    # Fallback: strip the aten:: prefix.
    return aten_op_name.split("::")[-1] if "::" in aten_op_name else aten_op_name


def _detect_num_layers(classified_kernels: List[Dict]) -> int:
    """Detect number of transformer layers from shared expert entry frequencies.

    Uses GCD of all shared expert entry frequencies as the proxy for num_layers.
    Returns 1 if no shared expert entries are found (safe default).
    """
    shared_freqs = [
        e["statistics"]["frequency"]
        for e in classified_kernels
        if e.get("expert_type") == "shared_expert"
        and e.get("statistics", {}).get("frequency", 0) > 0
    ]
    if not shared_freqs:
        return 1
    result = shared_freqs[0]
    for f in shared_freqs[1:]:
        result = math.gcd(result, f)
    return max(1, result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_shape(shape) -> List[int]:
    """Normalize possibly nested shape containers into a flat int list."""
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
    """Return the integer product of a shape list; 0 for empty."""
    dims = _normalize_shape(shape)
    if not dims:
        return 0
    result = 1
    for d in dims:
        result *= int(d)
    return result


def _ops_per_layer(freq: int, num_layers: int) -> int:
    """Return ops per layer; 0 if freq is not evenly divisible by num_layers."""
    if num_layers <= 0 or freq <= 0:
        return 0
    if freq % num_layers == 0:
        return freq // num_layers
    return 0


def _make_record(
    layer_id: int,
    op_name: str,
    hbm_fields: Dict,
    cta_count: int,
    latency_us: float,
    is_shared: bool,
    expert_type: str,
    structural_role: str,
    observed: bool,
    reconstruction_source: str,
    source_entry_id: Optional[str] = None,
    template_alias: Optional[str] = None,
) -> Dict:
    """Build a single op_profile record dict."""
    shared_expert_bytes = hbm_fields["hbm_bytes"] if is_shared else 0.0
    return {
        "layer_id": layer_id,
        "op_name": op_name,
        "flops": hbm_fields["flops"],
        "hbm_bytes": hbm_fields["hbm_bytes"],
        "weight_bytes": hbm_fields["weight_bytes"],
        "activation_bytes": hbm_fields["activation_bytes"],
        "kv_bytes": hbm_fields["kv_bytes"],
        "shared_expert_bytes": shared_expert_bytes,
        "cta_count": cta_count,
        "latency_us": latency_us,
        "is_shared_expert": is_shared,
        "expert_type": expert_type,
        "structural_role": structural_role,
        "observed": observed,
        "reconstruction_source": reconstruction_source,
        "source_entry_id": source_entry_id,
        "template_alias": template_alias,
    }


def _record_signature(r: Dict) -> Tuple:
    """Signature used to dedupe equivalent structural records."""
    return (
        r.get("structural_role"),
        int(r.get("flops", 0)),
        float(r.get("weight_bytes", 0.0)),
        float(r.get("activation_bytes", 0.0)),
        int(r.get("cta_count", 0)),
        round(float(r.get("latency_us", 0.0)), 6),
    )


def _select_representatives(records: List[Dict], target_count: int) -> List[Dict]:
    """Pick stable representatives from possibly duplicated observed records."""
    if not records or target_count <= 0:
        return []
    buckets: Dict[Tuple, List[Dict]] = {}
    for r in records:
        sig = _record_signature(r)
        buckets.setdefault(sig, []).append(r)
    ranked = sorted(
        buckets.values(),
        key=lambda rs: (-len(rs), float(rs[0].get("latency_us", 0.0))),
    )
    reps = [rs[0] for rs in ranked[:target_count]]
    while reps and len(reps) < target_count:
        reps.append(dict(reps[-1]))
    return reps


def _clone_record_with_alias(
    base: Dict,
    op_name: str,
    template_alias: str,
    observed: bool,
    reconstruction_source: str,
    structural_role: Optional[str] = None,
) -> Dict:
    out = dict(base)
    out["op_name"] = op_name
    out["template_alias"] = template_alias
    out["observed"] = observed
    out["reconstruction_source"] = reconstruction_source
    if structural_role is not None:
        out["structural_role"] = structural_role
    return out


def _global_canonical_by_role(records: List[Dict], role: str) -> Optional[Dict]:
    candidates = [
        r for r in records
        if r.get("layer_id", -1) >= 0 and r.get("structural_role") == role and r.get("observed")
    ]
    reps = _select_representatives(candidates, 1)
    return reps[0] if reps else None


def _reconstruct_shared_expert_template(
    records: List[Dict],
    num_layers: int,
) -> List[Dict]:
    """Reconstruct per-layer shared expert template into semantic aliases."""
    EXPECTED_SHARED_TEMPLATE = {
        "shared_expert_expand": 2,
        "shared_expert_down": 1,
    }
    shared_roles = set(EXPECTED_SHARED_TEMPLATE.keys())

    passthrough = [r for r in records if r.get("structural_role") not in shared_roles]
    shared = [r for r in records if r.get("structural_role") in shared_roles and r.get("layer_id", -1) >= 0]

    per_layer: Dict[int, Dict[str, List[Dict]]] = {
        layer_id: {"shared_expert_expand": [], "shared_expert_down": []}
        for layer_id in range(num_layers)
    }
    for r in shared:
        layer_id = int(r.get("layer_id", -1))
        if layer_id in per_layer:
            per_layer[layer_id][r.get("structural_role", "other")].append(r)

    global_expand = _global_canonical_by_role(shared, "shared_expert_expand")
    global_down = _global_canonical_by_role(shared, "shared_expert_down")

    reconstructed: List[Dict] = []
    for layer_id in range(num_layers):
        expands_obs = _select_representatives(per_layer[layer_id]["shared_expert_expand"], 2)
        downs_obs = _select_representatives(per_layer[layer_id]["shared_expert_down"], 1)

        synth_expands = 0
        synth_downs = 0

        if len(expands_obs) >= 2:
            expands = expands_obs[:2]
        elif len(expands_obs) == 1:
            expands = [expands_obs[0], dict(expands_obs[0])]
            synth_expands += 1
        else:
            expands = []
            if global_expand is not None:
                expands = [dict(global_expand), dict(global_expand)]
                synth_expands += 2
            elif downs_obs:
                expands = [dict(downs_obs[0]), dict(downs_obs[0])]
                synth_expands += 2

        if downs_obs:
            down = downs_obs[0]
        elif global_down is not None:
            down = dict(global_down)
            synth_downs += 1
        elif expands:
            down = dict(expands[0])
            synth_downs += 1
        else:
            down = None

        print(
            f"[moe.reconstruct] layer={layer_id} observed_expands={len(expands_obs)} "
            f"observed_downs={len(downs_obs)} synthesized_expands={synth_expands} "
            f"synthesized_downs={synth_downs}"
        )

        if len(expands) < 2 or down is None:
            continue

        gate_obs = bool(expands_obs)
        up_obs = len(expands_obs) >= 2
        down_obs = bool(downs_obs)

        gate_source = "trace" if gate_obs else "template_from_matching_role"
        up_source = "trace" if up_obs else "template_from_matching_role"
        down_source = "trace" if down_obs else "template_from_matching_role"

        gate_struct = dict(expands[0])
        gate_struct["layer_id"] = layer_id
        gate_struct["structural_role"] = "shared_expert_expand"
        reconstructed.append(_clone_record_with_alias(
            base=gate_struct,
            op_name="shared_expert_gate_proj",
            template_alias="gate_proj",
            observed=gate_obs,
            reconstruction_source=gate_source,
            structural_role="shared_expert_expand",
        ))

        up_struct = dict(expands[1])
        up_struct["layer_id"] = layer_id
        up_struct["structural_role"] = "shared_expert_expand"
        reconstructed.append(_clone_record_with_alias(
            base=up_struct,
            op_name="shared_expert_up_proj",
            template_alias="up_proj",
            observed=up_obs,
            reconstruction_source=up_source,
            structural_role="shared_expert_expand",
        ))

        down_struct = dict(down)
        down_struct["layer_id"] = layer_id
        down_struct["structural_role"] = "shared_expert_down"
        reconstructed.append(_clone_record_with_alias(
            base=down_struct,
            op_name="shared_expert_down_proj",
            template_alias="down_proj",
            observed=down_obs,
            reconstruction_source=down_source,
            structural_role="shared_expert_down",
        ))

    return passthrough + reconstructed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_op_profile(
    classified_kernels: List[Dict],
    num_layers: int,
    precision: str = "bfloat16",
    ncu_results: Optional[Dict[str, Dict]] = None,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """Generate per-layer per-op records for all kernels.

    Each kernel DB entry that fires an integer multiple of num_layers times
    is expanded into num_layers records (layer_id = 0 … num_layers-1).
    Entries with frequencies that are not divisible by num_layers, or that
    belong to the routed_expert type, get a single record with layer_id = -1.

    Shared expert kernels are flagged with is_shared_expert=True and
    shared_expert_bytes populated.

    Args:
        classified_kernels: Entries annotated with expert_type from
            classify_kernel_entries().
        num_layers: Number of transformer layers.
        precision:  Data type precision string (e.g., "bfloat16").
        ncu_results: Optional kernel_id -> NCU result dict; overrides
            shape-estimated hbm_bytes with actual dram_read + dram_write.
        output_path: If provided, writes op_profile.json to this path.

    Returns:
        List of record dicts, sorted by layer_id ASC (layer_id=-1 at end),
        then by op_name.
    """
    if not classified_kernels:
        records: List[Dict] = []
        if output_path is not None:
            Path(output_path).write_text(json.dumps(records, indent=2))
        return records

    dtype_bytes = _dtype_bytes(precision)
    ncu_results = ncu_results or {}
    num_layers = max(1, int(num_layers))

    raw_records = []

    for entry in classified_kernels:
        expert_type = entry.get("expert_type", "other")
        structural_role = entry.get("structural_role", "other")
        aten_op = entry.get("aten_op", {})
        aten_op_name = aten_op.get("name", "")
        input_dims = aten_op.get("input_dims", [])
        kernel = entry.get("kernel", {})
        stats = entry.get("statistics", {})
        entry_id = entry.get("id", "")

        freq = int(stats.get("frequency", 1))
        latency_us = float(stats.get("avg_duration_us", 0.0) or 0.0)

        # CTA count from kernel grid dims.
        grid = kernel.get("grid") or [1, 1, 1]
        cta_count = 1
        for dim in grid:
            cta_count *= int(dim) if dim else 1

        # Shape-based HBM field estimates.
        hbm_fields = _compute_hbm_fields(aten_op_name, input_dims, dtype_bytes)

        # NCU override: replace hbm_bytes with actual DRAM counters if available.
        if entry_id in ncu_results:
            ncu = ncu_results[entry_id]
            ncu_hbm = float(
                (ncu.get("hbm_read_bytes") or 0) + (ncu.get("hbm_write_bytes") or 0)
            )
            if ncu_hbm > 0:
                hbm_fields = dict(hbm_fields)
                hbm_fields["hbm_bytes"] = ncu_hbm

        is_shared = expert_type == "shared_expert"

        # Determine layer expansion: routed experts and non-divisible frequencies → layer_id=-1.
        ops_count = _ops_per_layer(freq, num_layers) if expert_type != "routed_expert" else 0
        is_layer_local = ops_count > 0

        if is_layer_local:
            for layer_id in range(num_layers):
                for _ in range(ops_count):
                    op_name_n = _infer_structural_op_name(aten_op_name, structural_role, input_dims)
                    raw_records.append(_make_record(
                        layer_id=layer_id,
                        op_name=op_name_n,
                        hbm_fields=hbm_fields,
                        cta_count=cta_count,
                        latency_us=latency_us,
                        is_shared=is_shared,
                        expert_type=expert_type,
                        structural_role=structural_role,
                        observed=True,
                        reconstruction_source="trace",
                        source_entry_id=entry_id,
                    ))
        else:
            op_name = _infer_structural_op_name(aten_op_name, structural_role, input_dims)
            raw_records.append(_make_record(
                layer_id=-1,
                op_name=op_name,
                hbm_fields=hbm_fields,
                cta_count=cta_count,
                latency_us=latency_us,
                is_shared=is_shared,
                expert_type=expert_type,
                structural_role=structural_role,
                observed=True,
                reconstruction_source="trace",
                source_entry_id=entry_id,
            ))

    records = _reconstruct_shared_expert_template(raw_records, num_layers)

    # Sort: layer_id ASC with -1 at the end, then op_name for stability.
    records.sort(key=lambda r: (
        r["layer_id"] if r["layer_id"] >= 0 else 10 ** 9,
        r["op_name"],
    ))

    if output_path is not None:
        Path(output_path).write_text(json.dumps(records, indent=2))

    return records
