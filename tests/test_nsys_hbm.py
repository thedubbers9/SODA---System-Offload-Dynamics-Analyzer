"""Unit tests for Nsight Systems sampled HBM attribution (no GPU / no nsys CLI)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("torch")

from soda.common.intervals import attribute_window_to_intervals, compute_overlap_ns
from soda.nsys import (
    attribute_windows_to_intervals,
    classify_dram_bandwidth_metrics,
    distribute_interval_bytes_to_soda_events,
    run_nsys_hbm_attribution,
    samples_to_byte_windows,
)


def test_compute_overlap_ns():
    assert compute_overlap_ns(0, 100, 50, 150) == 50
    assert compute_overlap_ns(0, 100, 100, 200) == 0
    assert compute_overlap_ns(10, 20, 15, 25) == 5


def test_attribute_window_to_intervals_split():
    intervals = [
        {"start_ns": 0, "end_ns": 250},
        {"start_ns": 250, "end_ns": 1000},
    ]
    shares, un = attribute_window_to_intervals(0, 1000, intervals)
    assert un == 0.0
    assert len(shares) == 2
    idx_to_sh = dict(shares)
    assert abs(idx_to_sh[0] - 0.25) < 1e-6
    assert abs(idx_to_sh[1] - 0.75) < 1e-6


def test_dram_metric_classification_variants():
    names = ["Foo", "Device 0 DRAM Read Bandwidth (%)", "other_metric DRAM Write Bandwidth"]
    r, w, sem = classify_dram_bandwidth_metrics(names)
    assert r is not None and "read" in r.lower()
    assert w is not None and "write" in w.lower()


def test_dram_metric_classification_gb20x_gpu_memory_labels():
    """Nsight GB20x configs often expose throughput as GPU Memory R/W, not 'Device … DRAM'."""
    names = [
        "Foo",
        "GPU Memory Read Throughput",
        "GPU Memory Write Throughput",
    ]
    r, w, sem = classify_dram_bandwidth_metrics(names)
    assert r == "GPU Memory Read Throughput"
    assert w == "GPU Memory Write Throughput"


def test_dram_metric_classification_nsys2025_throughput_suffix():
    """SQLite export uses TARGET_INFO_GPU_METRICS names like 'DRAM Read Bandwidth [Throughput %]'."""
    names = ["GPC Clock Frequency [MHz]", "DRAM Read Bandwidth [Throughput %]", "DRAM Write Bandwidth [Throughput %]"]
    r, w, _sem = classify_dram_bandwidth_metrics(names)
    assert r == "DRAM Read Bandwidth [Throughput %]"
    assert w == "DRAM Write Bandwidth [Throughput %]"


def test_samples_to_byte_windows_and_conservation():
    by_name = {
        "DRAM Read Bandwidth": [(0, 100.0), (1_000_000, 100.0)],
        "DRAM Write Bandwidth": [(0, 0.0), (1_000_000, 0.0)],
    }
    windows, kind = samples_to_byte_windows(
        by_name,
        "DRAM Read Bandwidth",
        "DRAM Write Bandwidth",
        peak_hbm_gbps=100.0,
        metric_kind="percent",
    )
    assert kind == "percent"
    integrated = sum(w["total_bytes"] for w in windows)
    intervals = [
        {"start_ns": 0, "end_ns": 500_000, "name": "k0", "correlation": 1},
        {"start_ns": 500_000, "end_ns": 2_000_000, "name": "k1", "correlation": 2},
    ]
    interval_attrs, ur, uw, ut = attribute_windows_to_intervals(windows, intervals)
    on_gpu = sum(float(x["nsys_hbm_estimated_total_bytes"]) for x in interval_attrs)
    assert abs(integrated - (on_gpu + ut)) / max(integrated, 1.0) < 1e-6


def _make_synthetic_sqlite(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE synth_gpu_metrics (
        event_timestamp INTEGER,
        value REAL,
        metric_name TEXT
    )"""
    )
    for ts, val, name in [
        (0, 80.0, "DRAM Read Bandwidth (%)"),
        (10_000, 80.0, "DRAM Read Bandwidth (%)"),
        (0, 20.0, "DRAM Write Bandwidth (%)"),
        (10_000, 20.0, "DRAM Write Bandwidth (%)"),
    ]:
        conn.execute(
            "INSERT INTO synth_gpu_metrics VALUES (?,?,?)",
            (ts, val, name),
        )
    conn.execute(
        """CREATE TABLE synth_cuda (
        start_ns INTEGER,
        end_ns INTEGER,
        kernel_name TEXT,
        correlationId INTEGER
    )"""
    )
    conn.execute(
        "INSERT INTO synth_cuda VALUES (?,?,?,?)",
        (0, 5000, "my_kernel_a", 101),
    )
    conn.execute(
        "INSERT INTO synth_cuda VALUES (?,?,?,?)",
        (5000, 10_000, "my_kernel_b", 102),
    )
    conn.commit()
    conn.close()


def test_end_to_end_sqlite_and_trace(tmp_path: Path):
    db = tmp_path / "prof.sqlite"
    _make_synthetic_sqlite(db)
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "kernel",
                "name": "my_kernel_a",
                "ts": 0.0,
                "dur": 5.0,
                "args": {"correlation": 101},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "my_kernel_b",
                "ts": 5.0,
                "dur": 5.0,
                "args": {"correlation": 102},
            },
        ]
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    out = tmp_path / "nsys_hbm"
    summ = run_nsys_hbm_attribution(
        db,
        trace_path,
        out,
        peak_hbm_gbps=100.0,
        read_metric_override="DRAM Read Bandwidth (%)",
        write_metric_override="DRAM Write Bandwidth (%)",
    )
    assert summ["num_windows"] >= 1
    assert summ["relative_error"] < 1e-5
    assert (out / "attribution_summary.json").is_file()
    traced = json.loads(trace_path.read_text(encoding="utf-8"))
    names = []
    for ev in traced["traceEvents"]:
        if ev.get("cat") == "kernel":
            args = ev.get("args") or {}
            names.append((ev.get("name"), args.get("nsys_hbm_estimated_total_bytes")))
    assert len(names) == 2
    b0, b1 = float(names[0][1]), float(names[1][1])
    assert b0 > 0 and b1 > 0
    assert abs(b0 - b1) / max(b0, b1, 1.0) < 0.6


def test_distribute_trace_unassigned(tmp_path: Path):
    """Bytes on Nsight intervals that do not overlap any trace GPU span -> trace_unassigned."""
    intervals = [
        {
            "start_ns": 0,
            "end_ns": 1000,
            "name": "empty",
            "correlation": None,
            "nsys_hbm_estimated_read_bytes": 0.0,
            "nsys_hbm_estimated_write_bytes": 0.0,
            "nsys_hbm_estimated_total_bytes": 0.0,
            "nsys_hbm_attribution_method": "sample_overlap",
            "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
        },
        {
            "start_ns": 1_000_000_000_000,
            "end_ns": 1_000_000_001_000,
            "name": "far",
            "correlation": None,
            "nsys_hbm_estimated_read_bytes": 50.0,
            "nsys_hbm_estimated_write_bytes": 0.0,
            "nsys_hbm_estimated_total_bytes": 50.0,
            "nsys_hbm_attribution_method": "sample_overlap",
            "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
        },
    ]
    soda = [
        {"ts_us": 0.0, "dur_us": 1.0, "name": "early", "correlation": None},
    ]
    match, ur, uw, ut, _, *_ = distribute_interval_bytes_to_soda_events(intervals, soda)
    assert ut > 0
    assert match[0]["nsys_hbm_match_status"] == "unmatched"


def test_distribute_min_start_alignment_when_no_correlation():
    """Without correlation IDs, offset 0 leaves timelines disjoint; min-start aligns them."""
    intervals = [
        {
            "start_ns": 0,
            "end_ns": 1000,
            "name": "lonely",
            "correlation": None,
            "nsys_hbm_estimated_read_bytes": 50.0,
            "nsys_hbm_estimated_write_bytes": 0.0,
            "nsys_hbm_estimated_total_bytes": 50.0,
            "nsys_hbm_attribution_method": "sample_overlap",
            "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
        }
    ]
    soda = [
        {"ts_us": 1_000_000.0, "dur_us": 1.0, "name": "late", "correlation": None},
    ]
    match, ur, uw, ut, _, *_ = distribute_interval_bytes_to_soda_events(intervals, soda)
    assert match[0]["nsys_hbm_match_status"] == "matched"
    assert float(match[0]["nsys_hbm_estimated_total_bytes"] or 0) > 0
    assert ut == 0.0
