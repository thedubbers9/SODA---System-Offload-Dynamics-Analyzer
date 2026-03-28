"""
Nsight Systems GPU metrics → HBM byte attribution (sampled device bandwidth).

Stage 1 option ``--nsys-hbm``: profile the benchmark subprocess with ``nsys profile``,
export ``.nsys-rep`` to SQLite, infer per-kernel HBM bytes via temporal overlap attribution.
"""
from __future__ import annotations

import csv
import json
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from soda.common.intervals import attribute_window_to_intervals, compute_overlap_ns

LOGGER = logging.getLogger("soda.nsys")

# Chrome / PyTorch profiler uses microseconds for ``ts`` and ``dur``.
US_TO_NS = 1000


def verify_nsys_available(nsys_bin: str = "nsys") -> None:
    """Raise RuntimeError with a clear message if ``nsys`` is not invokable."""
    from shutil import which

    if which(nsys_bin) is None and not Path(nsys_bin).is_file():
        raise RuntimeError(
            f"Nsight Systems CLI not found ({nsys_bin!r}). Install CUDA toolkit / Nsight Systems "
            f"or pass a valid path via --nsys-bin."
        )


def run_subprocess(cmd: List[str], cwd: Optional[Path] = None) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if r.returncode != 0:
        msg = r.stderr or r.stdout or "unknown error"
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\n{msg}")


def run_nsys_profile(
    nsys_bin: str,
    profile_args: List[str],
    python_executable: str,
    child_argv: List[str],
    cwd: Optional[Path] = None,
) -> None:
    """Run ``nsys profile`` wrapping ``python ... child_argv``."""
    cmd = [nsys_bin, "profile"] + profile_args + [python_executable, "-m", "soda"] + child_argv
    run_subprocess(cmd, cwd=cwd)


def export_nsys_sqlite(
    nsys_bin: str,
    rep_path: Path,
    sqlite_path: Path,
    *,
    force: bool = False,
) -> Path:
    """
    Export an ``.nsys-rep`` report to SQLite.

    Uses ``ns sys export --type sqlite`` (compatible with common Nsight Systems releases).
    """
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists() and force:
        sqlite_path.unlink()

    cmd = [
        nsys_bin,
        "export",
        "--type",
        "sqlite",
        "--output",
        str(sqlite_path),
        "--force-overwrite",
        "true",
        str(rep_path),
    ]
    try:
        run_subprocess(cmd)
    except RuntimeError:
        # Older nsys may omit force-overwrite or use different spelling
        cmd_fallback = [nsys_bin, "export", "--type", "sqlite", "--output", str(sqlite_path), str(rep_path)]
        if sqlite_path.exists():
            sqlite_path.unlink()
        run_subprocess(cmd_fallback)

    if not sqlite_path.is_file():
        raise RuntimeError(f"SQLite export did not produce {sqlite_path}")
    return sqlite_path


def open_sqlite(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return {str(r[1]): str(r[2]) for r in cur.fetchall()}


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    return [r[0] for r in cur.fetchall()]


def discover_gpu_metrics_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Heuristic discovery of GPU metric sample storage."""
    tables = list_tables(conn)
    string_ids_table = _find_string_ids_table(conn, tables)
    candidates = []
    for t in tables:
        cols = _table_columns(conn, t)
        col_lower = {c.lower(): c for c in cols}
        score = 0
        ts_col = None
        val_col = None
        name_col = None
        metric_id_col = None
        gpu_col = None
        for pattern, pts in (
            ("timestamp", 3),
            ("timestamps", 3),
            ("start", 2),
            ("time", 1),
        ):
            for c, orig in col_lower.items():
                if pattern in c and "duration" not in c:
                    ts_col = orig
                    score += pts
                    break
        for c, orig in col_lower.items():
            if c in ("value", "values", "metricvalue", "samplevalue"):
                val_col = orig
                score += 3
                break
        for c, orig in col_lower.items():
            if "metric" in c and "id" in c:
                metric_id_col = orig
                score += 2
                break
        for c, orig in col_lower.items():
            if c in ("textid", "metricid", "id") and metric_id_col is None:
                metric_id_col = orig
        for c, orig in col_lower.items():
            if "name" in c or c == "text" or "demangled" in c:
                name_col = orig
                score += 1
                break
        for c, orig in col_lower.items():
            if "gpu" in c and "id" in c:
                gpu_col = orig
                score += 1
                break
        if ts_col and val_col:
            candidates.append(
                {
                    "table": t,
                    "score": score,
                    "timestamp_col": ts_col,
                    "value_col": val_col,
                    "metric_id_col": metric_id_col,
                    "name_col": name_col,
                    "gpu_col": gpu_col,
                    "string_ids_table": string_ids_table,
                }
            )
    candidates.sort(key=lambda x: -x["score"])
    return {"tables": tables, "metric_candidates": candidates, "string_ids_table": string_ids_table}


def _find_string_ids_table(conn: sqlite3.Connection, tables: List[str]) -> Optional[str]:
    for t in tables:
        if "stringid" in t.lower():
            cols = _table_columns(conn, t)
            if any(c.lower() in ("id", "value") or "text" in c.lower() for c in cols):
                return t
    return None


def _resolve_string(conn: sqlite3.Connection, string_ids_table: Optional[str], sid: Any) -> str:
    if sid is None or string_ids_table is None:
        return ""
    try:
        cur = conn.execute(
            f'SELECT * FROM "{string_ids_table}" WHERE id = ? LIMIT 1', (int(sid),)
        )
        row = cur.fetchone()
        if row is None:
            return ""
        d = dict(row)
        for k in ("value", "text", "name", "str"):
            if k in d and d[k]:
                return str(d[k])
        return str(list(d.values())[-1] if d else "")
    except Exception:
        return ""


def load_gpu_metric_samples(
    conn: sqlite3.Connection,
    *,
    gpu_id: Optional[int] = None,
    schema_hint: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    disc = schema_hint or discover_gpu_metrics_schema(conn)
    cands = disc.get("metric_candidates") or []
    if not cands:
        raise RuntimeError(
            "Could not find GPU metric sample table in Nsight SQLite export. "
            "Ensure the report was captured with GPU Metrics enabled."
        )
    best = cands[0]
    t = best["table"]
    ts_c = best["timestamp_col"]
    val_c = best["value_col"]
    mid_c = best.get("metric_id_col")
    gpu_c = best.get("gpu_col")
    name_c = best.get("name_col")
    sid_tab = best.get("string_ids_table") or disc.get("string_ids_table")

    sel = [f'"{ts_c}" AS ts_raw', f'"{val_c}" AS val_raw']
    if mid_c:
        sel.append(f'"{mid_c}" AS metric_id')
    if name_c:
        sel.append(f'"{name_c}" AS name_inline')
    if gpu_c:
        sel.append(f'"{gpu_c}" AS gpu_id')
    q = f'SELECT {", ".join(sel)} FROM "{t}"'
    if gpu_id is not None and gpu_c:
        q += f' WHERE "{gpu_c}" = {int(gpu_id)}'

    rows = []
    for r in conn.execute(q):
        d = dict(r)
        ts = int(float(d["ts_raw"]))
        val = float(d["val_raw"])
        mid = d.get("metric_id")
        label = (d.get("name_inline") or "").strip()
        if not label and mid is not None:
            label = _resolve_string(conn, sid_tab, mid)
        gid = d.get("gpu_id")
        rows.append(
            {
                "timestamp_ns": ts,
                "value": val,
                "metric_name": label or f"id:{mid}",
                "metric_id": mid,
                "gpu_id": gid,
            }
        )
    rows.sort(key=lambda x: (x["timestamp_ns"], x["metric_name"]))
    return rows


def discover_cuda_kernel_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    tables = list_tables(conn)
    candidates = []
    skip = {"stringids", "sqlite_sequence"}
    for t in tables:
        tl = t.lower()
        if any(x in tl for x in skip):
            continue
        cols = _table_columns(conn, t)
        cl = {c.lower(): c for c in cols}
        # Need interval + label
        start_keys = [c for c in cols if any(x in c.lower() for x in ("start", "begin")) and "global" not in c.lower()]
        end_keys = [c for c in cols if any(x in c.lower() for x in ("end", "stop", "complete"))]
        if not start_keys:
            continue
        start_col = start_keys[0]
        end_col = end_keys[0] if end_keys else None
        name_col = None
        for pref in ("demangled", "name", "short", "text", "symbol"):
            for c in cols:
                if pref in c.lower():
                    name_col = c
                    break
            if name_col:
                break
        corr_col = None
        for c in cols:
            lc = c.lower()
            if "correlation" in lc or lc == "corrid":
                corr_col = c
                break
        stream_col = None
        for c in cols:
            if "stream" in c.lower():
                stream_col = c
                break
        kind = "unknown"
        if "kernel" in tl or "cupti" in tl or "cuda" in tl:
            kind = "cuda"
        score = 1
        if end_col:
            score += 3
        if name_col:
            score += 2
        if corr_col:
            score += 2
        if kind == "cuda" or "runtime" in tl:
            score += 2
        if score >= 3:
            candidates.append(
                {
                    "table": t,
                    "score": score,
                    "start_col": start_col,
                    "end_col": end_col,
                    "name_col": name_col,
                    "corr_col": corr_col,
                    "stream_col": stream_col,
                    "kind": kind,
                }
            )
    candidates.sort(key=lambda x: -x["score"])
    return {"candidates": candidates}


def _duration_fallback(conn: sqlite3.Connection, table: str, cols: Dict[str, str]) -> Optional[str]:
    for c in cols:
        if c.lower() in ("duration", "deltalatency", "latency", "timespan"):
            return c
    return None


def load_gpu_execution_intervals(
    conn: sqlite3.Connection,
    *,
    gpu_id: Optional[int] = None,
    schema_hint: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Load GPU kernel / device execution intervals from Nsight SQLite."""
    disc = schema_hint or discover_cuda_kernel_schema(conn)
    cands = disc.get("candidates") or []
    if not cands:
        return []

    intervals: List[Dict[str, Any]] = []
    seen = set()
    for cand in cands[:8]:  # merge a few top tables (memcpys may live separately)
        t = cand["table"]
        sc = cand["start_col"]
        ec = cand.get("end_col")
        cols = _table_columns(conn, t)
        if not ec:
            ec = _duration_fallback(conn, t, cols)
        if not ec:
            continue
        name_c = cand.get("name_col")
        corr_c = cand.get("corr_col")
        stream_c = cand.get("stream_col")
        gpu_c = None
        for c in cols:
            if "gpu" in c.lower() and "id" in c.lower():
                gpu_c = c
                break

        sel = [f'"{sc}" AS s_raw']
        if ec.lower() == sc.lower():
            continue
        el = ec.lower()
        if any(x in el for x in ("duration", "deltalatency", "latency", "timespan")) or (
            "delta" in el and "time" in el
        ):
            sel.append(f'("{sc}" + COALESCE("{ec}", 0)) AS e_raw')
        else:
            sel.append(f'"{ec}" AS e_raw')
        if name_c:
            sel.append(f'"{name_c}" AS name_raw')
        if corr_c:
            sel.append(f'"{corr_c}" AS corr_raw')
        if stream_c:
            sel.append(f'"{stream_c}" AS stream_raw')
        if gpu_c:
            sel.append(f'"{gpu_c}" AS gpu_raw')
        q = f'SELECT {", ".join(sel)} FROM "{t}"'
        if gpu_id is not None and gpu_c:
            q += f' WHERE "{gpu_c}" = {int(gpu_id)}'

        try:
            for r in conn.execute(q):
                d = dict(r)
                s = int(float(d["s_raw"]))
                e = int(float(d["e_raw"]))
                if e <= s:
                    continue
                nm = str(d.get("name_raw") or "")
                ck = (s, e, nm, t)
                if ck in seen:
                    continue
                seen.add(ck)
                corr = d.get("corr_raw")
                try:
                    corr_i = int(corr) if corr is not None else None
                except (TypeError, ValueError):
                    corr_i = None
                intervals.append(
                    {
                        "start_ns": s,
                        "end_ns": e,
                        "name": nm,
                        "correlation": corr_i,
                        "stream": d.get("stream_raw"),
                        "gpu_id": d.get("gpu_raw"),
                        "kind": "kernel" if "memcpy" not in nm.lower() and "memset" not in nm.lower() else "memcpy",
                        "source_table": t,
                    }
                )
        except sqlite3.Error as ex:
            LOGGER.warning("Skipping table %s: %s", t, ex)

    intervals.sort(key=lambda x: (x["start_ns"], x["end_ns"]))
    return intervals


# --- DRAM metric naming (robust, not one fixed string) ---

_EXACT_READ = frozenset(
    {
        "dram read bandwidth",
        "dram read throughput",
        "hbm read bandwidth",
    }
)
_EXACT_WRITE = frozenset(
    {
        "dram write bandwidth",
        "dram write throughput",
        "hbm write bandwidth",
    }
)


def classify_dram_bandwidth_metrics(
    metric_names: Sequence[str],
) -> Tuple[Optional[str], Optional[str], Dict[str, str]]:
    """
    Pick DRAM read / write bandwidth metric keys from a set of raw names.

    Returns:
        (read_name_or_none, write_name_or_none, semantic_map raw_name -> dram_read_bw|dram_write_bw)
    """
    semantic: Dict[str, str] = {}
    read_pick = None
    write_pick = None
    lower_map = {m: m.lower() for m in metric_names}

    for m, low in lower_map.items():
        if low in _EXACT_READ:
            read_pick = m
            semantic[m] = "dram_read_bw"
        elif low in _EXACT_WRITE:
            write_pick = m
            semantic[m] = "dram_write_bw"

    if read_pick is None:
        for m, low in lower_map.items():
            if "dram" in low and "read" in low and "bandwidth" in low:
                read_pick = m
                semantic[m] = "dram_read_bw"
                break
    if write_pick is None:
        for m, low in lower_map.items():
            if "dram" in low and "write" in low and "bandwidth" in low:
                write_pick = m
                semantic[m] = "dram_write_bw"
                break

    return read_pick, write_pick, semantic


def infer_metric_value_kind(
    sample_values: Sequence[float],
    metric_names_lower: Sequence[str],
) -> str:
    """
    Classify sampled metric values.

    Returns one of: ``percent``, ``bytes_per_sec``, ``unknown``.
    """
    joined = " ".join(metric_names_lower)
    if "percent" in joined or "%" in joined or "pct" in joined or "of peak" in joined:
        return "percent"
    if sample_values:
        mx = max(abs(v) for v in sample_values if v == v)
        if mx <= 100.0 + 1e-3:
            # Typical Nsight DRAM bandwidth is % of peak (0-100+)
            return "percent"
        if mx > 1e6:
            return "bytes_per_sec"
    return "unknown"


def samples_to_byte_windows(
    samples_by_metric: Dict[str, List[Tuple[int, float]]],
    read_metric: Optional[str],
    write_metric: Optional[str],
    *,
    peak_hbm_gbps: Optional[float],
    metric_kind: str,
    last_window_ns: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build [t_i, t_{i+1}) windows; integrate bandwidth into bytes per window.

    ``samples_by_metric`` maps metric name -> sorted list of (timestamp_ns, value).

    Returns:
        (windows, resolved_metric_kind) where resolved may normalize unknown->percent with peak.
    """
    if not read_metric and not write_metric:
        raise ValueError("Need at least one DRAM bandwidth metric")

    ts_union: List[int] = sorted(
        {t for lst in samples_by_metric.values() for t, _ in lst}
    )
    if len(ts_union) < 2 and last_window_ns is None:
        raise RuntimeError("Not enough GPU metric samples to form windows.")

    # Align read/write on union timestamps via last-known value
    def series_at(ts_list: List[Tuple[int, float]], t_query: int) -> float:
        if not ts_list:
            return 0.0
        v = 0.0
        last_t = -1
        for t, val in ts_list:
            if t <= t_query:
                v = val
                last_t = t
            else:
                break
        return v if last_t >= 0 else 0.0

    read_series = samples_by_metric.get(read_metric or "", [])
    write_series = samples_by_metric.get(write_metric or "", [])

    kind = metric_kind
    if kind == "unknown":
        if peak_hbm_gbps is None:
            raise RuntimeError(
                "Could not infer whether Nsight DRAM bandwidth metrics are percent-of-peak or B/s. "
                "Pass --gpu-peak-hbm-bandwidth-gbps to convert sampled metrics to bytes."
            )
        kind = "percent"

    peak_bps: Optional[float] = None
    if kind == "percent":
        if peak_hbm_gbps is None:
            raise RuntimeError(
                "Nsight Systems DRAM bandwidth metrics are reported as percent of peak sustained. "
                "Provide --gpu-peak-hbm-bandwidth-gbps to convert to bytes/sec."
            )
        peak_bps = float(peak_hbm_gbps) * 1e9

    def to_bps(val: float) -> float:
        if kind == "percent" and peak_bps is not None:
            return max(0.0, val) / 100.0 * peak_bps
        if kind == "bytes_per_sec":
            return max(0.0, val)
        raise RuntimeError(f"Unsupported metric kind for conversion: {kind}")

    windows: List[Dict[str, Any]] = []
    for i in range(len(ts_union) - 1):
        t0, t1 = ts_union[i], ts_union[i + 1]
        if t1 <= t0:
            continue
        rv = to_bps(series_at(read_series, t0))
        wv = to_bps(series_at(write_series, t0))
        dt_s = (t1 - t0) * 1e-9
        rb = rv * dt_s
        wb = wv * dt_s
        windows.append(
            {
                "start_ns": t0,
                "end_ns": t1,
                "read_bw_Bps": rv,
                "write_bw_Bps": wv,
                "read_bytes": rb,
                "write_bytes": wb,
                "total_bytes": rb + wb,
            }
        )

    if len(ts_union) >= 2:
        deltas = [ts_union[i + 1] - ts_union[i] for i in range(len(ts_union) - 1)]
        median_delta = sorted(deltas)[len(deltas) // 2]
    else:
        median_delta = int(last_window_ns or 10_000_000)

    if ts_union:
        t_last = ts_union[-1]
        t_end = t_last + median_delta
        rv = to_bps(series_at(read_series, t_last))
        wv = to_bps(series_at(write_series, t_last))
        dt_s = (t_end - t_last) * 1e-9
        rb = rv * dt_s
        wb = wv * dt_s
        windows.append(
            {
                "start_ns": t_last,
                "end_ns": t_end,
                "read_bw_Bps": rv,
                "write_bw_Bps": wv,
                "read_bytes": rb,
                "write_bytes": wb,
                "total_bytes": rb + wb,
            }
        )

    return windows, kind


def attribute_windows_to_intervals(
    windows: Sequence[Dict[str, Any]],
    intervals: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """
    For each window, split read/write/total bytes across overlapping GPU intervals.

    Returns:
        (interval_attrs, unassigned_read, unassigned_write, unassigned_total)
        interval_attrs align with ``intervals`` index order (same list passed in).
    """
    n = len(intervals)
    acc_read = [0.0] * n
    acc_write = [0.0] * n
    acc_tot = [0.0] * n
    ur = uw = ut = 0.0

    for w in windows:
        ws = int(w["start_ns"])
        we = int(w["end_ns"])
        rb = float(w["read_bytes"])
        wb = float(w["write_bytes"])
        tb = float(w["total_bytes"])

        shares, un_assigned = attribute_window_to_intervals(ws, we, list(intervals))

        for idx, sh in shares:
            acc_read[idx] += rb * sh
            acc_write[idx] += wb * sh
            acc_tot[idx] += tb * sh
        ur += rb * un_assigned
        uw += wb * un_assigned
        ut += tb * un_assigned

    out = []
    for i, it in enumerate(intervals):
        out.append(
            {
                **it,
                "nsys_hbm_estimated_read_bytes": acc_read[i],
                "nsys_hbm_estimated_write_bytes": acc_write[i],
                "nsys_hbm_estimated_total_bytes": acc_tot[i],
                "nsys_hbm_attribution_method": "sample_overlap",
                "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
            }
        )
    return out, ur, uw, ut


def _soda_gpu_events_from_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same ordering as MoE ``_iter_gpu_events_ordered`` (kernels + mem ops)."""
    out: List[Dict[str, Any]] = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat = event.get("cat", "")
        name = event.get("name", "")
        args = event.get("args", {}) or {}
        ts = float(event.get("ts", 0) or 0)
        dur = float(event.get("dur", 0) or 0)
        if cat == "kernel":
            out.append(
                {
                    "kind": "gpu_kernel",
                    "name": name,
                    "ts_us": ts,
                    "dur_us": dur,
                    "correlation": args.get("correlation"),
                    "stream": args.get("stream"),
                    "device": args.get("device"),
                }
            )
        elif cat in ("gpu_memcpy", "gpu_memset"):
            out.append(
                {
                    "kind": "gpu_memcpy",
                    "name": name,
                    "ts_us": ts,
                    "dur_us": dur,
                    "correlation": args.get("correlation"),
                    "stream": args.get("stream"),
                    "device": args.get("device"),
                }
            )
    return out


def inject_nsys_gpu_intervals_into_chrome_trace(
    sqlite_path: Path,
    trace_path: Path,
) -> int:
    """
    Append Chrome-trace GPU events from Nsight SQLite so ``trace.json`` contains
    ``kernel`` / ``gpu_memcpy`` rows after CPU-only PyTorch profiling under ``nsys``.

    PyTorch cannot use ``ProfilerActivity.CUDA`` while ``nsys profile`` owns the
    CUPTI subscription. Timestamps use the same ns→µs mapping as attribution
    (``ts`` = start_ns / 1000, ``dur`` = (end_ns - start_ns) / 1000).

    Returns:
        Number of events appended (0 if trace already had GPU X events).
    """
    trace_path = Path(trace_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    has_gpu = any(
        e.get("ph") == "X"
        and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        for e in events
    )
    if has_gpu:
        return 0

    conn = open_sqlite(Path(sqlite_path))
    try:
        intervals = load_gpu_execution_intervals(conn)
    finally:
        conn.close()

    new_events: List[Dict[str, Any]] = []
    for it in intervals:
        s_ns = int(it["start_ns"])
        e_ns = int(it["end_ns"])
        if e_ns <= s_ns:
            continue
        name = str(it.get("name") or "unknown")
        if it.get("kind") == "memcpy":
            cat = "gpu_memcpy"
            if "memset" in name.lower():
                cat = "gpu_memset"
        else:
            cat = "kernel"
        corr = it.get("correlation")
        args: Dict[str, Any] = {
            "nsys_injected": True,
            "correlation": corr,
            "stream": it.get("stream"),
            "device": it.get("gpu_id", 0),
        }
        new_events.append(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": 0,
                "tid": 0,
                "ts": s_ns / 1000.0,
                "dur": (e_ns - s_ns) / 1000.0,
                "args": args,
            }
        )

    trace.setdefault("traceEvents", []).extend(new_events)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)

    return len(new_events)


def _estimate_time_offset_ns(
    soda_events: Sequence[Dict[str, Any]],
    nsys_intervals: Sequence[Dict[str, Any]],
) -> int:
    """Median offset (nsys_start - soda_start) for correlation-matched pairs."""
    diffs: List[int] = []
    ns_by_corr: Dict[int, List[int]] = {}
    for it in nsys_intervals:
        c = it.get("correlation")
        if c is None:
            continue
        ns_by_corr.setdefault(int(c), []).append(int(it["start_ns"]))
    for ev in soda_events:
        c = ev.get("correlation")
        if c is None:
            continue
        lst = ns_by_corr.get(int(c))
        if not lst:
            continue
        soda_start_ns = int(ev["ts_us"] * US_TO_NS)
        for ns in lst[:1]:
            diffs.append(ns - soda_start_ns)
    if not diffs:
        return 0
    diffs.sort()
    return diffs[len(diffs) // 2]


def distribute_interval_bytes_to_soda_events(
    interval_attrs: Sequence[Dict[str, Any]],
    soda_events: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], float, float, float, int]:
    """
    For each Nsight interval, split its attributed bytes across overlapping SODA GPU
    events proportionally to overlap duration (same rule as sample windows).

    Returns:
        (soda_index -> field dict, trace_unassigned_read, trace_unassigned_write,
         trace_unassigned_total, time_offset_ns)
    """
    offset_ns = _estimate_time_offset_ns(soda_events, interval_attrs)
    n = len(soda_events)
    acc_read = [0.0] * n
    acc_write = [0.0] * n
    acc_tot = [0.0] * n
    ur = uw = ut = 0.0

    for it in interval_attrs:
        is_ = int(it["start_ns"])
        ie_ = int(it["end_ns"])
        rb = float(it.get("nsys_hbm_estimated_read_bytes", 0) or 0)
        wb = float(it.get("nsys_hbm_estimated_write_bytes", 0) or 0)
        tb = float(it.get("nsys_hbm_estimated_total_bytes", 0) or 0)
        overlaps: List[Tuple[int, int]] = []
        for j, ev in enumerate(soda_events):
            ss = int(float(ev["ts_us"]) * US_TO_NS) + offset_ns
            se = int(float(ev["ts_us"] + ev["dur_us"]) * US_TO_NS) + offset_ns
            ov = compute_overlap_ns(is_, ie_, ss, se)
            if ov > 0:
                overlaps.append((j, ov))
        denom = sum(o for _, o in overlaps)
        if denom <= 0:
            ur += rb
            uw += wb
            ut += tb
            continue
        for j, o in overlaps:
            sh = o / denom
            acc_read[j] += rb * sh
            acc_write[j] += wb * sh
            acc_tot[j] += tb * sh

    out: Dict[int, Dict[str, Any]] = {}
    for j in range(n):
        matched = acc_tot[j] > 0
        out[j] = {
            "nsys_hbm_estimated_read_bytes": acc_read[j] if matched else None,
            "nsys_hbm_estimated_write_bytes": acc_write[j] if matched else None,
            "nsys_hbm_estimated_total_bytes": acc_tot[j] if matched else None,
            "nsys_hbm_attribution_method": "sample_overlap",
            "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
            "nsys_hbm_match_status": "matched" if matched else "unmatched",
            "nsys_time_offset_ns_applied": offset_ns,
        }
    return out, ur, uw, ut, offset_ns


def patch_trace_json_with_nsys(
    trace_path: Path,
    soda_match: Dict[int, Dict[str, Any]],
) -> None:
    """Write nsys fields into matching trace event ``args`` (in file order)."""
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    idx = 0
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        m = soda_match.get(idx)
        idx += 1
        if not m:
            continue
        args = ev.setdefault("args", {})
        for k, v in m.items():
            if k not in args:
                args[k] = v

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)


def write_sample_windows_csv(path: Path, windows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not windows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "start_ns",
        "end_ns",
        "read_bw_Bps",
        "write_bw_Bps",
        "read_bytes",
        "write_bytes",
        "total_bytes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in windows:
            w.writerow(row)


def write_kernel_hbm_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_nsys_hbm_attribution(
    sqlite_path: Path,
    trace_path: Path,
    out_dir: Path,
    *,
    peak_hbm_gbps: Optional[float],
    gpu_id: Optional[int] = None,
    read_metric_override: Optional[str] = None,
    write_metric_override: Optional[str] = None,
    keep_raw: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end: SQLite → windows → interval attribution → soda kernel match → artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = open_sqlite(sqlite_path)
    try:
        mschema = discover_gpu_metrics_schema(conn)
        samples = load_gpu_metric_samples(conn, gpu_id=gpu_id, schema_hint=mschema)

        by_name: Dict[str, List[Tuple[int, float]]] = {}
        for s in samples:
            by_name.setdefault(s["metric_name"], []).append((s["timestamp_ns"], s["value"]))
        for k in by_name:
            by_name[k].sort(key=lambda x: x[0])

        names = list(by_name.keys())
        read_m, write_m, _sem = classify_dram_bandwidth_metrics(names)
        if read_metric_override:
            read_m = read_metric_override
        if write_metric_override:
            write_m = write_metric_override

        if not read_m and not write_m:
            conn.close()
            raise RuntimeError(
                f"No DRAM read/write bandwidth metrics found in SQLite. Metric names seen: {names[:30]}"
            )

        values_for_kind: List[float] = []
        for mn in [read_m, write_m]:
            if mn and mn in by_name:
                values_for_kind.extend(v for _, v in by_name[mn][:200])
        kind = infer_metric_value_kind(values_for_kind, [n.lower() for n in names if n])

        windows, resolved_kind = samples_to_byte_windows(
            by_name,
            read_m,
            write_m,
            peak_hbm_gbps=peak_hbm_gbps,
            metric_kind=kind,
        )
        write_sample_windows_csv(out_dir / "sample_windows.csv", windows)

        kschema = discover_cuda_kernel_schema(conn)
        intervals = load_gpu_execution_intervals(conn, gpu_id=gpu_id, schema_hint=kschema)
        conn.close()
    except Exception:
        conn.close()
        raise

    if keep_raw:
        with open(out_dir / "raw_gpu_metrics.json", "w", encoding="utf-8") as f:
            json.dump(samples[:50000], f, indent=2)
        write_kernel_hbm_csv(out_dir / "raw_cuda_intervals.csv", intervals)

    interval_attrs, ur_win, uw_win, ut_win = attribute_windows_to_intervals(windows, intervals)
    write_kernel_hbm_csv(out_dir / "kernel_hbm_attribution.csv", interval_attrs)

    interval_total_bytes = sum(float(it.get("nsys_hbm_estimated_total_bytes", 0) or 0) for it in interval_attrs)

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    soda_ev = _soda_gpu_events_from_trace(trace)
    soda_match, ur_tr, uw_tr, ut_tr, _off = distribute_interval_bytes_to_soda_events(interval_attrs, soda_ev)

    unmatched_rows: List[Dict[str, Any]] = []
    for it in interval_attrs:
        is_ = int(it["start_ns"])
        ie_ = int(it["end_ns"])
        hit = False
        for ev in soda_ev:
            ss = int(float(ev["ts_us"]) * US_TO_NS) + _off
            se = int(float(ev["ts_us"] + ev["dur_us"]) * US_TO_NS) + _off
            if compute_overlap_ns(is_, ie_, ss, se) > 0:
                hit = True
                break
        if not hit:
            unmatched_rows.append({k: v for k, v in it.items() if not k.startswith("nsys_hbm_")})
    write_kernel_hbm_csv(out_dir / "unmatched_intervals.csv", unmatched_rows)

    patch_trace_json_with_nsys(trace_path, soda_match)

    integrated = sum(float(w["total_bytes"]) for w in windows)
    interval_level_attr = interval_total_bytes + ut_win
    denom_int = max(integrated, 1.0)
    relative_window_interval = abs(integrated - interval_level_attr) / denom_int

    soda_attributed_total = sum(
        float(soda_match[j].get("nsys_hbm_estimated_total_bytes") or 0) for j in soda_match
    )
    trace_level = soda_attributed_total + ut_tr
    denom_tr = max(interval_total_bytes, 1.0)
    relative_trace = abs(interval_total_bytes - trace_level) / denom_tr

    summary = {
        "integrated_total_bytes": integrated,
        "window_unassigned_read_bytes": ur_win,
        "window_unassigned_write_bytes": uw_win,
        "window_unassigned_total_bytes": ut_win,
        "interval_total_bytes": interval_total_bytes,
        "interval_level_conservation_error": relative_window_interval,
        "trace_attributed_total_bytes": soda_attributed_total,
        "trace_unassigned_read_bytes": ur_tr,
        "trace_unassigned_write_bytes": uw_tr,
        "trace_unassigned_total_bytes": ut_tr,
        "trace_level_conservation_error": relative_trace,
        "relative_error": relative_window_interval,
        "num_windows": len(windows),
        "num_gpu_intervals": len(interval_attrs),
        "num_matched_kernel_records": sum(
            1 for v in soda_match.values() if v.get("nsys_hbm_match_status") == "matched"
        ),
        "metric_units": f"{resolved_kind}->bytes_per_sec",
        "dram_read_metric": read_m,
        "dram_write_metric": write_m,
        "peak_hbm_bandwidth_gbps_used": peak_hbm_gbps,
    }
    with open(out_dir / "attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = {
        "sqlite_path": str(sqlite_path.resolve()),
        "trace_path": str(trace_path.resolve()),
        "metric_discovery": mschema.get("metric_candidates", [])[:3],
        "kernel_discovery": kschema.get("candidates", [])[:5],
    }
    with open(out_dir / "nsys_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return summary


def strip_nsys_parent_argv(argv: List[str]) -> List[str]:
    """Remove parent-only flags (and their values) before spawning the child profiler."""
    skip_next = 0
    out: List[str] = []
    i = 0
    parent_flags = {
        "--nsys-hbm",
        "--nsys-force-overwrite",
        "--nsys-keep-intermediate",
        "--gpu-peak-hbm-bandwidth-gbps",
        "--nsys-bin",
        "--nsys-output",
        "--nsys-gpu-metrics-devices",
        "--nsys-gpu-metrics-set",
        "--nsys-gpu-metrics-frequency",
        "--nsys-sqlite",
    }
    while i < len(argv):
        if skip_next:
            skip_next -= 1
            i += 1
            continue
        tok = argv[i]
        if "=" in tok:
            key, _val = tok.split("=", 1)
            if key in parent_flags:
                i += 1
                continue
        if tok in parent_flags:
            if tok == "--nsys-hbm":
                i += 1
                continue
            skip_next = 1
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def top_kernels_by_nsys_bytes(
    trace_path: Path,
    k: int = 20,
) -> List[Dict[str, Any]]:
    """Scan patched trace for top GPU events by ``nsys_hbm_estimated_total_bytes``."""
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    scored: List[Tuple[float, str]] = []
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = ev.get("args") or {}
        b = args.get("nsys_hbm_estimated_total_bytes")
        if b is None:
            continue
        try:
            bf = float(b)
        except (TypeError, ValueError):
            continue
        scored.append((bf, ev.get("name", "")))
    scored.sort(reverse=True)
    return [{"name": n, "nsys_hbm_estimated_total_bytes": v} for v, n in scored[:k]]
