"""
GPU power sampling via NVML (Stage 1 profiling only).

NVML reports **board-level** power per GPU (mW). Per-kernel / per-op **energy**
is estimated by integrating that curve over each kernel's GPU execution interval
in the profiler trace (linear trace-time → wall-time mapping). This is
**attributed** energy, not a hardware per-kernel power measurement.

Optional: total energy counters (millijoules) when supported by the driver/GPU.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_NVML = None
_NVML_IMPORT_ERROR: Optional[str] = None


def _try_load_nvml() -> Tuple[Any, Optional[str]]:
    global _NVML, _NVML_IMPORT_ERROR
    if _NVML is not None:
        return _NVML, _NVML_IMPORT_ERROR
    try:
        import pynvml as nvml  # type: ignore

        _NVML = nvml
        _NVML_IMPORT_ERROR = None
        return nvml, None
    except ImportError as e:
        _NVML_IMPORT_ERROR = str(e)
        return None, _NVML_IMPORT_ERROR


def nvml_available() -> bool:
    nvml, err = _try_load_nvml()
    if nvml is None:
        return False
    try:
        nvml.nvmlInit()
        nvml.nvmlShutdown()
        return True
    except Exception:
        return False


def nvml_import_error() -> Optional[str]:
    _, err = _try_load_nvml()
    return err


@dataclass
class PowerSample:
    """One timestamped multi-GPU snapshot."""

    t_rel_ms: float
    devices: List[Dict[str, Any]]


@dataclass
class PowerSessionResult:
    """Result of sampling over a profiling window."""

    interval_ms: float
    duration_ms: float
    sample_count: int
    devices: List[Dict[str, Any]]
    samples: List[Dict[str, Any]] = field(default_factory=list)
    energy_counter_mj: Optional[List[Dict[str, Any]]] = None


class PowerSampler:
    """
    Background NVML poller; call start() before work and stop() after.

    Thread-safe; stores samples with monotonic timestamps relative to start.
    """

    def __init__(self, num_devices: int, interval_ms: float = 5.0):
        self.num_devices = num_devices
        self.interval_ms = max(0.5, float(interval_ms))
        self.interval_s = self.interval_ms / 1000.0
        self._nvml, _ = _try_load_nvml()
        self._handles: List[Any] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples: List[PowerSample] = []
        self._t0_ns: int = 0
        self._energy_start: Optional[List[Optional[int]]] = None
        self._energy_end: Optional[List[Optional[int]]] = None
        self._error: Optional[str] = None

    def _read_energy_mj(self, handle) -> Optional[int]:
        if self._nvml is None:
            return None
        try:
            return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        except Exception:
            return None

    def _sample_devices(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, h in enumerate(self._handles):
            row: Dict[str, Any] = {"index": idx}
            try:
                row["power_mw"] = int(self._nvml.nvmlDeviceGetPowerUsage(h))
            except Exception as e:
                row["power_mw"] = None
                row["power_error"] = str(e)
            for name, clock_type in (
                ("sm_clock_mhz", self._nvml.NVML_CLOCK_SM),
                ("mem_clock_mhz", self._nvml.NVML_CLOCK_MEM),
            ):
                try:
                    row[name] = int(self._nvml.nvmlDeviceGetClockInfo(h, clock_type))
                except Exception:
                    row[name] = None
            try:
                row["power_limit_mw"] = int(self._nvml.nvmlDeviceGetEnforcedPowerLimit(h))
            except Exception:
                row["power_limit_mw"] = None
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                row["gpu_util_pct"] = int(util.gpu)
                row["mem_util_pct"] = int(util.memory)
            except Exception:
                row["gpu_util_pct"] = None
                row["mem_util_pct"] = None
            out.append(row)
        return out

    def _loop(self) -> None:
        assert self._nvml is not None
        while not self._stop.is_set():
            t_ns = time.monotonic_ns()
            rel_ms = (t_ns - self._t0_ns) / 1e6
            self._samples.append(PowerSample(t_rel_ms=rel_ms, devices=self._sample_devices()))
            if self._stop.wait(self.interval_s):
                break

    def start(self) -> bool:
        if self._nvml is None:
            self._error = nvml_import_error() or "pynvml not available"
            return False
        try:
            self._nvml.nvmlInit()
        except Exception as e:
            self._error = f"nvmlInit: {e}"
            return False
        self._handles = []
        self._energy_start = []
        try:
            for i in range(self.num_devices):
                h = self._nvml.nvmlDeviceGetHandleByIndex(i)
                self._handles.append(h)
                self._energy_start.append(self._read_energy_mj(h))
        except Exception as e:
            self._error = f"nvmlDeviceGetHandleByIndex: {e}"
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            return False

        self._samples = []
        self._stop.clear()
        self._t0_ns = time.monotonic_ns()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> Optional[PowerSessionResult]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30.0)
            self._thread = None

        if self._nvml is None:
            return None

        self._energy_end = []
        for h in self._handles:
            self._energy_end.append(self._read_energy_mj(h))

        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

        if self._error:
            return None

        t1_ns = time.monotonic_ns()
        duration_ms = (t1_ns - self._t0_ns) / 1e6

        device_summaries = _aggregate_samples(self._samples, self.num_devices)
        energy_counter = _energy_delta_rows(self._energy_start, self._energy_end)

        sample_dicts = [
            {"t_rel_ms": round(s.t_rel_ms, 4), "devices": s.devices} for s in self._samples
        ]

        return PowerSessionResult(
            interval_ms=self.interval_ms,
            duration_ms=round(duration_ms, 4),
            sample_count=len(self._samples),
            devices=device_summaries,
            samples=sample_dicts,
            energy_counter_mj=energy_counter,
        )


def _energy_delta_rows(
    start: Optional[List[Optional[int]]],
    end: Optional[List[Optional[int]]],
) -> Optional[List[Dict[str, Any]]]:
    if not start or not end or len(start) != len(end):
        return None
    rows = []
    for i, (a, b) in enumerate(zip(start, end)):
        if a is None or b is None:
            rows.append({"index": i, "delta_mj": None, "note": "counter_unavailable"})
        else:
            d = b - a
            if d < 0:
                rows.append({"index": i, "delta_mj": d, "note": "counter_wrap_or_reset"})
            else:
                rows.append({"index": i, "delta_mj": d})
    return rows


def _integrate_device_power_mj(samples: List[PowerSample], powers: List[Optional[float]]) -> float:
    """Trapezoidal integration of power (mW) over sample timestamps → millijoules."""
    pts: List[Tuple[float, float]] = []
    for s, p in zip(samples, powers):
        if p is not None:
            pts.append((s.t_rel_ms, float(p)))
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        dt_s = (pts[i + 1][0] - pts[i][0]) / 1000.0
        if dt_s <= 0:
            continue
        total += 0.5 * (pts[i][1] + pts[i + 1][1]) * dt_s
    return total


def _aggregate_samples(samples: List[PowerSample], num_devices: int) -> List[Dict[str, Any]]:
    if not samples:
        return []
    per_dev: List[List[Optional[float]]] = [[] for _ in range(num_devices)]
    for s in samples:
        by_idx = {int(d["index"]): d.get("power_mw") for d in s.devices if "index" in d}
        for idx in range(num_devices):
            v = by_idx.get(idx)
            per_dev[idx].append(float(v) if v is not None else None)

    out: List[Dict[str, Any]] = []
    for idx in range(num_devices):
        series = per_dev[idx]
        vals = [v for v in series if v is not None]
        if not vals:
            out.append({"index": idx, "power_mw": {}})
            continue
        avg = sum(vals) / len(vals)
        mj = _integrate_device_power_mj(samples, series)
        out.append(
            {
                "index": idx,
                "power_mw": {
                    "avg": round(avg, 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "samples": len(vals),
                },
                "estimated_energy_mj_trapezoid": round(mj, 4),
            }
        )
    return out


def build_power_metrics_dict(result: PowerSessionResult) -> Dict[str, Any]:
    """Compact summary for report.json (no raw timeseries)."""
    total_mj_trap = 0.0
    for d in result.devices:
        e = d.get("estimated_energy_mj_trapezoid")
        if isinstance(e, (int, float)):
            total_mj_trap += float(e)
    out: Dict[str, Any] = {
        "source": "nvml",
        "interval_ms": result.interval_ms,
        "window_duration_ms": result.duration_ms,
        "sample_count": result.sample_count,
        "per_device": result.devices,
        "total_estimated_energy_mj_trapezoid_sum_devices": round(total_mj_trap, 4),
    }
    if result.energy_counter_mj:
        out["hardware_energy_counter_mj"] = result.energy_counter_mj
    return out


def _interp_p(t0: float, p0: float, t1: float, p1: float, t: float) -> float:
    if t1 == t0:
        return p0
    return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


def _device_power_series(samples: List[Dict[str, Any]], device_index: int) -> List[Tuple[float, float]]:
    """Build (t_rel_ms, power_mw) with forward-fill for missing samples."""
    out: List[Tuple[float, float]] = []
    last_p: Optional[float] = None
    for s in samples:
        t = float(s["t_rel_ms"])
        p: Optional[float] = None
        for d in s.get("devices", []):
            if int(d.get("index", -1)) == device_index:
                v = d.get("power_mw")
                if v is not None:
                    p = float(v)
                break
        if p is not None:
            last_p = p
        if last_p is not None:
            out.append((t, last_p))
    return out


def integrate_power_interval_mj(
    samples: List[Dict[str, Any]],
    device_index: int,
    t_lo_ms: float,
    t_hi_ms: float,
) -> float:
    """
    Trapezoidal integral of NVML power (mW) for one GPU over [t_lo_ms, t_hi_ms]
    on the same time axis as ``t_rel_ms`` in power samples (NVML window start = 0).
    """
    if t_hi_ms <= t_lo_ms:
        return 0.0
    pts = _device_power_series(samples, device_index)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(len(pts) - 1):
        t0, p0 = pts[i]
        t1, p1 = pts[i + 1]
        seg_lo = max(t0, t_lo_ms)
        seg_hi = min(t1, t_hi_ms)
        if seg_hi <= seg_lo:
            continue
        pa = _interp_p(t0, p0, t1, p1, seg_lo)
        pb = _interp_p(t0, p0, t1, p1, seg_hi)
        dt_s = (seg_hi - seg_lo) / 1000.0
        total += 0.5 * (pa + pb) * dt_s
    return total


def _kernel_interval_nvml_ms(
    ts_us: float,
    dur_us: float,
    trace_min_us: float,
    trace_max_us: float,
    gap_ms: float,
    profile_wall_ms: float,
) -> Tuple[float, float]:
    """Map profiler kernel GPU interval (Chrome ``ts``/``dur``) to NVML sample time (ms)."""
    span_us = trace_max_us - trace_min_us
    span_ms = span_us / 1000.0
    if span_ms <= 0:
        return gap_ms, gap_ms + profile_wall_ms
    scale = profile_wall_ms / span_ms
    start_us = ts_us
    end_us = ts_us + max(0.0, dur_us)
    rel_start_ms = (start_us - trace_min_us) / 1000.0 * scale
    rel_end_ms = (end_us - trace_min_us) / 1000.0 * scale
    return gap_ms + rel_start_ms, gap_ms + rel_end_ms


def attribute_nvml_power_to_trace(
    events: Dict[str, Any],
    sequences: List[Dict[str, Any]],
    power_samples: Optional[List[Dict[str, Any]]],
    nvml_mono_start_ns: int,
    profile_mono_start_ns: int,
    profile_mono_end_ns: int,
    num_runs: int,
    export_instances: bool,
) -> Dict[str, Any]:
    """
    Attribute board power energy (mJ) to each kernel instance and aggregate by
    kernel name / ATen op name.

    Returns a dict suitable for JSON; may be merged into ``power_metrics``.
    """
    if not power_samples:
        return {"available": False, "reason": "no_power_timeseries"}

    kernels = events.get("gpu", {}).get("kernels") or []
    if not kernels:
        return {"available": False, "reason": "no_gpu_kernels_in_trace"}

    gap_ms = (profile_mono_start_ns - nvml_mono_start_ns) / 1e6
    profile_wall_ms = max(0.0, (profile_mono_end_ns - profile_mono_start_ns) / 1e6)

    trace_min_us = min(k["ts"] for k in kernels)
    trace_max_us = max(k["ts"] + float(k.get("dur", 0) or 0) for k in kernels)

    from soda.common.data import clean_kernel_name

    seq_by_kid: Dict[int, Dict[str, Any]] = {}
    for seq in sequences:
        kern = seq.get("kernel")
        if kern is not None:
            seq_by_kid[id(kern)] = seq

    energy_by_kernel_id: Dict[int, float] = {}
    instances: List[Dict[str, Any]] = []

    for k in kernels:
        dev = int(k.get("device") if k.get("device") is not None else 0)
        t_lo, t_hi = _kernel_interval_nvml_ms(
            float(k["ts"]),
            float(k.get("dur", 0) or 0),
            trace_min_us,
            trace_max_us,
            gap_ms,
            profile_wall_ms,
        )
        mj = integrate_power_interval_mj(power_samples, dev, t_lo, t_hi)
        energy_by_kernel_id[id(k)] = mj
        if export_instances:
            raw_name = k.get("name", "")
            sq = seq_by_kid.get(id(k))
            aten_n = None
            torch_n = None
            if sq:
                ao = sq.get("aten_op") or {}
                aten_n = ao.get("name")
                to = sq.get("torch_op") or {}
                torch_n = to.get("name")
            instances.append(
                {
                    "kernel_name": raw_name,
                    "kernel_name_clean": clean_kernel_name(raw_name),
                    "ts_us": k["ts"],
                    "dur_us": k.get("dur", 0),
                    "device": dev,
                    "attributed_energy_mj": round(mj, 6),
                    "nvml_interval_ms": [round(t_lo, 4), round(t_hi, 4)],
                    "aten_op_name": aten_n,
                    "torch_op_name": torch_n,
                }
            )

    by_kernel: Dict[str, Dict[str, Any]] = {}
    for k in kernels:
        raw_name = k.get("name", "")
        ck = clean_kernel_name(raw_name)
        mj = energy_by_kernel_id[id(k)]
        if ck not in by_kernel:
            by_kernel[ck] = {"invocations": 0, "total_energy_mj": 0.0}
        by_kernel[ck]["invocations"] += 1
        by_kernel[ck]["total_energy_mj"] += mj

    for v in by_kernel.values():
        inv = max(1, int(v["invocations"]))
        v["total_energy_mj"] = round(v["total_energy_mj"], 6)
        v["avg_energy_mj_per_invocation"] = round(v["total_energy_mj"] / inv, 8)

    by_aten: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"invocations": 0, "total_energy_mj": 0.0})
    for seq in sequences:
        kern = seq.get("kernel")
        aten = seq.get("aten_op")
        if not kern or not aten:
            continue
        mj = energy_by_kernel_id.get(id(kern))
        if mj is None:
            continue
        name = aten.get("name", "unknown")
        by_aten[name]["invocations"] += 1
        by_aten[name]["total_energy_mj"] += mj

    for name, v in by_aten.items():
        inv = max(1, int(v["invocations"]))
        v["total_energy_mj"] = round(v["total_energy_mj"], 6)
        v["avg_energy_mj_per_invocation"] = round(v["total_energy_mj"] / inv, 8)

    by_torch: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"invocations": 0, "total_energy_mj": 0.0})
    for seq in sequences:
        kern = seq.get("kernel")
        to = seq.get("torch_op")
        if not kern or not to:
            continue
        mj = energy_by_kernel_id.get(id(kern))
        if mj is None:
            continue
        tname = to.get("name", "unknown")
        by_torch[tname]["invocations"] += 1
        by_torch[tname]["total_energy_mj"] += mj

    for name, v in by_torch.items():
        inv = max(1, int(v["invocations"]))
        v["total_energy_mj"] = round(v["total_energy_mj"], 6)
        v["avg_energy_mj_per_invocation"] = round(v["total_energy_mj"] / inv, 8)

    total_trace_mj = sum(energy_by_kernel_id.values())
    nr = max(1, int(num_runs or 1))

    out: Dict[str, Any] = {
        "available": True,
        "methodology": (
            "NVML board power (mW) integrated over each kernel's GPU execution interval "
            "in the Chrome trace; trace timestamps are linearly mapped to wall time between "
            "the torch.profiler region start/end. Attributed energy is not a hardware "
            "per-kernel counter."
        ),
        "mapping": {
            "gap_ms_nvml_to_profile_start": round(gap_ms, 4),
            "profile_wall_ms": round(profile_wall_ms, 4),
            "trace_kernel_span_us": [trace_min_us, trace_max_us],
        },
        "totals": {
            "attributed_energy_all_kernel_instances_mj": round(total_trace_mj, 6),
            "attributed_energy_per_profiled_run_mj": round(total_trace_mj / nr, 8),
            "num_profiled_runs": nr,
        },
        "by_kernel_name": dict(sorted(by_kernel.items(), key=lambda x: -x[1]["total_energy_mj"])),
        "by_aten_op_name": dict(sorted(by_aten.items(), key=lambda x: -x[1]["total_energy_mj"])),
        "by_torch_op_name": dict(sorted(by_torch.items(), key=lambda x: -x[1]["total_energy_mj"])),
    }
    if export_instances:
        out["kernel_instances"] = instances
    return out


@contextmanager
def power_profile_scope(tracer: Any, args: Any):
    """
    NVML power sampling for the profiled inference window (Stage 1).

    Sets on ``tracer``:
      - ``power_metrics``: summary dict (or unavailable reason)
      - ``_power_timeseries_full``: raw samples for power_timeseries.json
    """
    tracer.power_metrics = None
    tracer._power_timeseries_full = None
    if not getattr(tracer, "_has_cuda", False):
        yield
        return
    if getattr(args, "no_power_sampling", False):
        yield
        return
    interval = float(getattr(args, "power_interval_ms", 5.0))
    sampler = PowerSampler(num_devices=tracer.num_gpus, interval_ms=interval)
    if not sampler.start():
        tracer.power_metrics = {
            "available": False,
            "reason": sampler._error or "nvml_start_failed",
        }
        if nvml_import_error():
            tracer.power_metrics["hint"] = "pip install nvidia-ml-py"
        yield
        return
    tracer._nvml_mono_start_ns = sampler._t0_ns
    try:
        yield
    finally:
        result = sampler.stop()
        if result:
            tracer.power_metrics = build_power_metrics_dict(result)
            tracer._power_timeseries_full = result.samples
        else:
            tracer.power_metrics = {
                "available": False,
                "reason": getattr(sampler, "_error", None) or "nvml_stop_failed",
            }
