"""Trace-centric MoE op profile: launch↔kernel join, NCU attach, layer bucketing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soda.moe.op_profile import (
    _compute_layer_boundaries,
    build_execution_trace,
    generate_op_profile,
)


def _aten(name, ts, dur, ext_id, input_dims=None):
    return {
        "ph": "X",
        "cat": "cpu_op",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {
            "External id": ext_id,
            "Input Dims": input_dims or [],
            "Input type": [],
            "Input Strides": [],
            "Concrete Inputs": [],
        },
    }


def _launch(ts, dur, corr, ext_id):
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": "cudaLaunchKernel",
        "ts": ts,
        "dur": dur,
        "args": {"correlation": corr, "External id": ext_id},
    }


def _kernel(name, ts, dur, corr, ext_id):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": dur,
        "args": {
            "correlation": corr,
            "External id": ext_id,
            "grid": [2, 1, 1],
            "block": [128, 1, 1],
        },
    }


def _memcpy(ts, dur):
    return {
        "ph": "X",
        "cat": "gpu_memcpy",
        "name": "Memcpy DtoH",
        "ts": ts,
        "dur": dur,
        "args": {},
    }


@pytest.fixture
def classified_mm_entry():
    return [{
        "id": "K0001",
        "aten_op": {"name": "aten::mm", "input_dims": [[4, 8], [8, 16]]},
        "kernel": {"name": "sgemm_kernel", "raw_name": "sgemm_kernel"},
        "expert_type": "shared_expert",
        "structural_role": "shared_expert_expand",
        "statistics": {},
    }]


def _minimal_one_mm_trace():
    return {
        "traceEvents": [
            _aten("aten::mm", 100, 50, 1, [[4, 8], [8, 16]]),
            _launch(140, 5, 10, 1),
            _kernel("sgemm_kernel", 200, 30, 10, 1),
        ]
    }


def test_launch_kernel_correlation_join(tmp_path, classified_mm_entry):
    trace_path = tmp_path / "t.json"
    trace_path.write_text(json.dumps(_minimal_one_mm_trace()))
    rows = build_execution_trace(
        trace_path, classified_mm_entry, num_layers=1, precision="bfloat16"
    )
    assert len(rows) == 1
    assert rows[0]["kind"] == "gpu_kernel"
    assert rows[0]["cuda_launch"] is not None
    assert rows[0]["cuda_launch"]["correlation"] == 10
    assert rows[0]["aten_op"]["name"] == "aten::mm"
    assert rows[0]["cta_count"] == 2


def test_ncu_attaches_after_trace(tmp_path, classified_mm_entry):
    trace_path = tmp_path / "t.json"
    trace_path.write_text(json.dumps(_minimal_one_mm_trace()))
    rows = build_execution_trace(
        trace_path,
        classified_mm_entry,
        num_layers=1,
        ncu_results={"K0001": {"hbm_read_bytes": 100.0, "hbm_write_bytes": 50.0}},
    )
    assert rows[0]["HBM_byte_data_from_ncu"] is True
    assert rows[0]["hbm_bytes"] == pytest.approx(150.0)


def test_memcpy_classified_as_copy(tmp_path, classified_mm_entry):
    trace = {
        "traceEvents": [
            _aten("aten::mm", 100, 40, 1, [[4, 8], [8, 16]]),
            _launch(130, 4, 7, 1),
            _kernel("sgemm_kernel", 180, 20, 7, 1),
            _memcpy(250, 3),
        ],
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    rows = build_execution_trace(p, classified_mm_entry, num_layers=1)
    assert len(rows) == 2
    assert rows[1]["kind"] == "gpu_memcpy"
    assert rows[1]["semantic_role"] == "copy / other"


def test_layer_boundaries_from_norm_pattern():
    aten_ordered = []
    for i in range(4):
        aten_ordered.append({
            "name": "aten::rms_norm",
            "ts": float(i * 10),
            "dur": 2.0,
            "external_id": i + 1,
            "aten_index": i,
            "input_dims": [],
        })
    starts = _compute_layer_boundaries(aten_ordered, num_layers=2)
    assert starts[0] == 0
    assert starts[1] == 2


def test_generate_op_profile_writes_files(tmp_path, classified_mm_entry):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(_minimal_one_mm_trace()))
    out = tmp_path / "moe_profile" / "op_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    et = tmp_path / "moe_profile" / "execution_trace.json"
    prof = generate_op_profile(
        trace_path=trace_path,
        classified_kernels=classified_mm_entry,
        num_layers=1,
        output_path=out,
        execution_trace_path=et,
        moe_debug_log_path=None,
    )
    assert prof["row_count"] == 1
    assert et.is_file()
    assert out.is_file()
    disk = json.loads(out.read_text())
    assert disk["schema_version"] == 2
    assert "aggregates" in disk
    assert disk["row_count"] == 1
