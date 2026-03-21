"""
Sanity-check MoE auto-detection against Qwen1.5-MoE-A2.7B-style tensor geometry.

HF reference (Qwen1.5-MoE-A2.7B):
  hidden_dim=2048, shared_dim=5632, routed_dim=1408, num_experts=60
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from soda.moe.detect import (
    QWEN15_MOE_A27B_DIMS,
    classify_kernel_entries,
    detect_moe_config,
)


def _kernel_mm(
    act_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    *,
    frequency: int = 24,
    kernel_id: str = "k",
) -> Dict:
    """Minimal kernel DB entry: aten::mm with activation, weight shapes."""
    return {
        "id": kernel_id,
        "aten_op": {
            "name": "aten::mm",
            "input_dims": [list(act_shape), list(weight_shape)],
        },
        "kernel": {"grid": [1, 1, 1], "block": [128, 1, 1]},
        "statistics": {"frequency": frequency, "avg_duration_us": 1.0},
    }


def _kernel_linear(
    act_shape: Tuple[int, ...],
    weight_shape: Tuple[int, ...],
    *,
    frequency: int = 24,
    kernel_id: str = "k",
) -> Dict:
    """Minimal kernel DB entry: aten::linear (weight N x K)."""
    return {
        "id": kernel_id,
        "aten_op": {
            "name": "aten::linear",
            "input_dims": [list(act_shape), list(weight_shape)],
        },
        "kernel": {"grid": [1, 1, 1], "block": [128, 1, 1]},
        "statistics": {"frequency": frequency, "avg_duration_us": 1.0},
    }


def _qwen_like_kernel_list() -> List[Dict]:
    """Synthetic kernels mimicking Qwen1.5-MoE-A2.7B GEMM shapes (2D, low-card experts)."""
    h = QWEN15_MOE_A27B_DIMS["hidden_dim"]
    s = QWEN15_MOE_A27B_DIMS["shared_dim"]
    r = QWEN15_MOE_A27B_DIMS["routed_dim"]
    e = QWEN15_MOE_A27B_DIMS["num_experts"]
    layers = 24
    freq = layers  # divisible by num_layers for downstream op_profile

    kernels: List[Dict] = []

    # Gate: linear (out=e, in=h) — 2D activations so is_3d does not force "attention".
    kernels.append(
        _kernel_linear((128, h), (e, h), frequency=freq, kernel_id="gate")
    )

    # Shared expert expand: mm (h -> s)
    kernels.append(_kernel_mm((128, h), (h, s), frequency=freq, kernel_id="sh_exp_a"))
    kernels.append(_kernel_mm((128, h), (h, s), frequency=freq, kernel_id="sh_exp_b"))

    # Shared expert down: mm (s -> h)
    kernels.append(_kernel_mm((128, s), (s, h), frequency=freq, kernel_id="sh_down"))

    # Routed expert expand: high cardinality (many token counts), same weight
    for i, ntok in enumerate([32, 48, 64, 80, 96, 112], start=1):
        kernels.append(
            _kernel_mm((ntok, h), (h, r), frequency=1, kernel_id=f"rt_exp_{i}")
        )

    # Routed expert down: high cardinality
    for i, ntok in enumerate([32, 48, 64, 80, 96, 112], start=1):
        kernels.append(
            _kernel_mm((ntok, r), (r, h), frequency=1, kernel_id=f"rt_down_{i}")
        )

    return kernels


def test_detect_moe_config_qwen15_moe_a27b_like():
    kernels = _qwen_like_kernel_list()
    cfg = detect_moe_config(kernels)
    assert cfg["detection_method"] == "cardinality"
    assert cfg["hidden_dim"] == QWEN15_MOE_A27B_DIMS["hidden_dim"]
    assert cfg["shared_dim"] == QWEN15_MOE_A27B_DIMS["shared_dim"]
    assert cfg["routed_dim"] == QWEN15_MOE_A27B_DIMS["routed_dim"]
    assert cfg["num_experts"] == QWEN15_MOE_A27B_DIMS["num_experts"]


def test_classify_qwen15_shared_roles_expand_and_down():
    kernels = _qwen_like_kernel_list()
    classified = classify_kernel_entries(kernels)
    by_id = {e["id"]: e for e in classified}
    assert by_id["sh_exp_a"]["expert_type"] == "shared_expert"
    assert by_id["sh_exp_a"]["structural_role"] == "shared_expert_expand"
    assert by_id["sh_exp_b"]["structural_role"] == "shared_expert_expand"
    assert by_id["sh_down"]["expert_type"] == "shared_expert"
    assert by_id["sh_down"]["structural_role"] == "shared_expert_down"
    assert by_id["gate"]["structural_role"] == "moe_gate"
