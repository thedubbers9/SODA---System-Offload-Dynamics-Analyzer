"""
Nsight Systems GPU metrics → HBM byte attribution (sampled device bandwidth).

Stage 1 option ``--nsys-hbm``: profile the benchmark subprocess with ``nsys profile``,
export ``.nsys-rep`` to SQLite, infer per-kernel HBM bytes via temporal overlap attribution.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import subprocess
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from soda.common.intervals import compute_overlap_ns

LOGGER = logging.getLogger("soda.nsys")

# Chrome / PyTorch profiler uses microseconds for ``ts`` and ``dur``.
US_TO_NS = 1000

# Timeline bucketing for overlap queries (ns). Keeps attribution ~linear when intervals are short.
_NS_OVERLAP_CHUNK_NS = 1_000_000  # 1 ms


def _nsys_hbm_parallel_workers() -> int:
    """Use at most 2/3 of logical CPUs for heavy attribution passes."""
    c = os.cpu_count() or 8
    return max(1, (c * 2) // 3)

# Progress reporting: milestone prints every 10% for large loops (avoid terminal spam).
_NS_HBM_PROGRESS_MIN_STEPS = 50_000
_NS_HBM_INDEX_PROGRESS_MIN = 300_000
_NS_HBM_SAMPLE_FETCH_LOG_EVERY = 500_000


def _nsys_hbm_status(msg: str) -> None:
    print(f"[nsys-hbm] {msg}", flush=True)


class _NsysHbmMilestones:
    """At most ~11 lines per phase: start + 10%…100% when ``total`` is large."""

    __slots__ = ("_label", "_total", "_next_pct", "_on")

    def __init__(self, label: str, total: int, *, min_total: int = _NS_HBM_PROGRESS_MIN_STEPS) -> None:
        self._label = label
        self._total = total
        self._on = total >= min_total and total > 0
        self._next_pct = 10
        if self._on:
            print(f"[nsys-hbm] {label} — starting ({total:,} steps)", flush=True)

    def step(self, i_done: int) -> None:
        if not self._on:
            return
        pct = 100.0 * i_done / self._total
        while self._next_pct <= 100 and pct + 1e-9 >= self._next_pct:
            print(
                f"[nsys-hbm] {self._label} — {self._next_pct}% ({i_done:,}/{self._total:,})",
                flush=True,
            )
            self._next_pct += 10


def _series_last_value_at_or_before(ts_list: List[Tuple[int, float]], t_query: int) -> float:
    """``ts_list`` sorted by timestamp; return last sample value at or before ``t_query``."""
    if not ts_list:
        return 0.0
    times = [t for t, _ in ts_list]
    i = bisect_right(times, t_query) - 1
    if i < 0:
        return 0.0
    return float(ts_list[i][1])


def _build_ns_chunk_index(
    spans: Sequence[Tuple[int, int]],
    chunk_ns: int = _NS_OVERLAP_CHUNK_NS,
    *,
    progress_label: Optional[str] = None,
) -> Tuple[List[int], List[List[int]]]:
    """
    Bucket possibly-overlapping [start, end) spans into chunk ids for fast range lookup.

    Returns:
        (chunk_starts_sorted, chunk_to_indices) where ``chunk_to_indices[k]`` lists span
        indices whose span intersects chunk ``chunk_starts_sorted[k]``.
    """
    n_sp = len(spans)
    idx_label = progress_label if progress_label else "timeline index"
    prog = _NsysHbmMilestones(idx_label, n_sp, min_total=_NS_HBM_INDEX_PROGRESS_MIN)
    bucketed: Dict[int, List[int]] = defaultdict(list)
    for idx, (ss, se) in enumerate(spans):
        if se <= ss:
            prog.step(idx + 1)
            continue
        b0 = ss // chunk_ns
        b1 = (se - 1) // chunk_ns
        for b in range(b0, b1 + 1):
            bucketed[b].append(idx)
        prog.step(idx + 1)
    if not bucketed:
        return [], []
    chunks = sorted(bucketed.keys())
    lists = [bucketed[c] for c in chunks]
    return chunks, lists


def _span_indices_touching_interval(
    is_lo: int,
    ie_hi: int,
    chunk_starts: List[int],
    chunk_lists: List[List[int]],
    chunk_ns: int = _NS_OVERLAP_CHUNK_NS,
) -> List[int]:
    """Union of span indices that may intersect [is_lo, ie_hi) (caller still computes exact overlap)."""
    if ie_hi <= is_lo or not chunk_starts:
        return []
    c0 = is_lo // chunk_ns
    c1 = (ie_hi - 1) // chunk_ns
    i0 = bisect_left(chunk_starts, c0)
    i1 = bisect_right(chunk_starts, c1)
    if i0 >= i1:
        return []
    cand: set[int] = set()
    for ci in range(i0, i1):
        cand.update(chunk_lists[ci])
    return list(cand)


def _is_gpu_metrics_sample_table(table: str) -> bool:
    return table.replace("_", "").lower() == "gpumetrics"


def _nsys_can_use_tmpdir(tmp_root: Path) -> bool:
    """True if Nsight can create ``<tmp>/nvidia/nsight_systems`` (often fails under ``/tmp`` on shared systems)."""
    try:
        probe = tmp_root / "nvidia" / "nsight_systems"
        probe.mkdir(parents=True, exist_ok=True)
        t = probe / ".soda_write_probe"
        t.write_text("ok", encoding="utf-8")
        t.unlink()
        return True
    except OSError:
        return False


def subprocess_env_for_nsys() -> dict[str, str]:
    """
    Environment for ``nsys`` CLI invocations.

    If ``$TMPDIR``/``$TMP``/``/tmp`` cannot host ``nvidia/nsight_systems``, set ``TMPDIR`` to a
    user-writable cache dir (same remedy as NVIDIA documents for restricted ``/tmp``).
    """
    env = os.environ.copy()
    candidates: list[Path] = []
    for key in ("TMPDIR", "TMP"):
        v = env.get(key, "").strip()
        if v:
            candidates.append(Path(v).expanduser().resolve())
    candidates.append(Path("/tmp"))
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if _nsys_can_use_tmpdir(root):
            env["TMPDIR"] = str(root)
            return env
    fallback = Path.home() / ".cache" / "soda-nsight-systems"
    fallback.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(fallback)
    LOGGER.info("Using fallback TMPDIR for nsys: %s", fallback)
    return env


def verify_nsys_available(nsys_bin: str = "nsys") -> None:
    """Raise RuntimeError with a clear message if ``nsys`` is not invokable."""
    from shutil import which

    if which(nsys_bin) is None and not Path(nsys_bin).is_file():
        raise RuntimeError(
            f"Nsight Systems CLI not found ({nsys_bin!r}). Install CUDA toolkit / Nsight Systems "
            f"or pass a valid path via --nsys-bin."
        )


def run_subprocess(
    cmd: List[str],
    cwd: Optional[Path] = None,
    *,
    env: Optional[dict[str, str]] = None,
) -> None:
    LOGGER.info("Running: %s", " ".join(cmd))
    r = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=env,
    )
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
    run_subprocess(cmd, cwd=cwd, env=subprocess_env_for_nsys())


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
    nsys_env = subprocess_env_for_nsys()
    try:
        run_subprocess(cmd, env=nsys_env)
    except RuntimeError:
        # Older nsys may omit force-overwrite or use different spelling
        cmd_fallback = [nsys_bin, "export", "--type", "sqlite", "--output", str(sqlite_path), str(rep_path)]
        if sqlite_path.exists():
            sqlite_path.unlink()
        run_subprocess(cmd_fallback, env=nsys_env)

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
        tl = t.lower()
        # Never treat string-id / symbol tables as metric sample sources (wrong JOIN targets).
        if any(
            x in tl
            for x in (
                "stringid",
                "string_ids",
                "demangle",
                "symbol",
                "sqlite_sequence",
            )
        ):
            continue
        cols = _table_columns(conn, t)
        col_lower = {c.lower(): c for c in cols}
        score = 0
        ts_col = None
        val_col = None
        name_col = None
        metric_id_col = None
        gpu_col = None
        # Strong signal: Nsight GPU metrics tables (names vary by version / chip).
        tl_nounderscore = tl.replace("_", "")
        if "gpumetric" in tl or "gpu_metric" in tl or tl_nounderscore == "gpumetrics":
            score += 40
        if "gpu" in tl and "metric" in tl and "sample" in tl:
            score += 35
        if "gpu" in tl and "counter" in tl and "sample" in tl:
            score += 30
        if "nvtx" in tl or "osrt" in tl or "etw" in tl:
            score -= 40
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
            if ts_col:
                break
        if not ts_col:
            for key in ("ts", "tscode", "timens", "timestampsns", "starttime", "start_time"):
                if key in col_lower:
                    ts_col = col_lower[key]
                    score += 2
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
        if metric_id_col is None:
            for key in ("metricid", "metric_id", "textid", "text_id"):
                if key in col_lower:
                    metric_id_col = col_lower[key]
                    break
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


def _string_table_text_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cols = _table_columns(conn, table)
    preferred = []
    for pref in ("value", "text", "name", "str", "string"):
        for c in cols:
            lc = c.lower()
            if lc == pref or lc.endswith(pref) or pref in lc:
                if c not in preferred:
                    preferred.append(c)
    return preferred or list(cols.keys())


def _label_is_obviously_not_a_metric_name(s: str) -> bool:
    """Inline ``name`` column on some Nsight tables holds NVTX text, paths, or env blobs."""
    if not s:
        return True
    if len(s) > 400:
        return True
    low = s.lower()
    if s.startswith("/") and ("opt/" in low or "scratch/" in low or "bin/" in low):
        return True
    if "=" in s and len(s) > 100 and ("path=" in low or "conda_" in low or "microbench/" in low):
        return True
    return False


def _text_like_dram_hbm_bandwidth_metric(s: str) -> bool:
    """Rough filter aligned with ``classify_dram_bandwidth_metrics`` (for SQL / string-table scans)."""
    low = s.lower()
    if "read" not in low and "write" not in low:
        return False
    if not any(t in low for t in ("bandwidth", "throughput", "%", " bw", "bw ")):
        return False
    if "dram" in low or "hbm" in low:
        return True
    if "device" in low and "memory" in low:
        return True
    # e.g. "GPU Memory Read Throughput" on GB20x configs (no "DRAM" in the label).
    if "memory" in low and "host" not in low and "pageable" not in low and "pinned" not in low:
        return True
    return False


def _collect_dram_bandwidth_string_ids(
    conn: sqlite3.Connection, string_table: str, text_col: str
) -> Dict[int, str]:
    """Map string-id -> label for rows that look like device DRAM/HBM bandwidth metrics."""
    out: Dict[int, str] = {}
    try:
        q = f'''SELECT * FROM "{string_table}" WHERE "{text_col}" IS NOT NULL
            AND (
              LOWER(CAST("{text_col}" AS TEXT)) LIKE '%bandwidth%'
              OR LOWER(CAST("{text_col}" AS TEXT)) LIKE '%throughput%'
            )'''
        for r in conn.execute(q):
            d = dict(r)
            val = d.get(text_col)
            if val is None:
                continue
            text = str(val).strip()
            if not text or not _text_like_dram_hbm_bandwidth_metric(text):
                continue
            rid = _sqlite_row_int_id(r)
            if rid is None:
                continue
            out[rid] = text
    except sqlite3.Error:
        return {}
    return out


def _metric_reference_column_names(col_lower: Dict[str, str]) -> List[str]:
    """Column names that may hold a string-table id for GPU metric samples."""
    out: List[str] = []
    for key, orig in col_lower.items():
        if key in (
            "metricid",
            "metric_id",
            "textid",
            "text_id",
            "gpumetricid",
            "gpu_metric_id",
            "counterid",
            "counter_id",
            "metrictextid",
        ):
            out.append(orig)
        elif "metric" in key and "id" in key:
            out.append(orig)
    # Plain ``id`` only when the table is clearly GPU-metric sampling (avoid row PKs elsewhere).
    return list(dict.fromkeys(out))


def _is_likely_string_dictionary_table(conn: sqlite3.Connection, table: str) -> bool:
    stl = table.lower()
    if "stringid" in stl:
        return True
    cols = _table_columns(conn, table)
    cl = {c.lower(): typ for c, typ in cols.items()}
    if "id" not in cl:
        return False
    for c, typ in cols.items():
        if c.lower() == "id":
            continue
        t = str(typ).upper()
        if t.startswith(("TEXT", "VARCHAR", "CHAR")):
            return True
    return False


def _sqlite_row_int_id(row: Any) -> Optional[int]:
    d = dict(row)
    for k, v in d.items():
        if str(k).lower() == "id" and v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def _gpu_metric_name_catalog(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    """
    Nsight Systems 2025.x: sample rows live in ``GPU_METRICS`` with ``metricId`` + ``typeId``;
    human-readable names are in ``TARGET_INFO_GPU_METRICS`` (not the global ``StringIds`` table).
    """
    tables = list_tables(conn)
    cat_table = next((t for t in tables if t.upper() == "TARGET_INFO_GPU_METRICS"), None)
    if cat_table is None:
        return None
    cols = _table_columns(conn, cat_table)
    cl = {c.lower(): c for c in cols}
    if not all(k in cl for k in ("typeid", "metricid", "metricname")):
        return None
    return {
        "table": cat_table,
        "type_col": cl["typeid"],
        "metric_id_col": cl["metricid"],
        "name_col": cl["metricname"],
    }


def _catalog_dram_bandwidth_metric_ids(
    conn: sqlite3.Connection, cat: Dict[str, str]
) -> List[int]:
    """
    ``TARGET_INFO_GPU_METRICS`` is tiny; use it to find ``metricId`` values for DRAM/HBM
    read+write bandwidth so ``GPU_METRICS`` can be filtered in SQL instead of loading every counter.
    """
    ct = cat["table"]
    mic = cat["metric_id_col"]
    nc = cat["name_col"]
    out: List[int] = []
    try:
        for row in conn.execute(f'SELECT "{mic}", "{nc}" FROM "{ct}"'):
            mid, mname = row[0], row[1]
            if mid is None or mname is None:
                continue
            label = str(mname).strip()
            if not label:
                continue
            read_p, write_p, _ = classify_dram_bandwidth_metrics([label])
            if read_p == label or write_p == label:
                out.append(int(mid))
    except sqlite3.Error:
        return []
    return list(dict.fromkeys(out))


def find_gpu_dram_metric_sample_binding(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """
    Locate the GPU metric sample table by tying string-table rows (DRAM/HBM bandwidth names)
    to a numeric/text id column. Heuristic table scoring alone is unreliable on Nsight 2025.x /
    GB20x exports where a generic ``name`` column holds NVTX noise.
    """
    if _gpu_metric_name_catalog(conn) is not None:
        # StringIds row IDs are unrelated to ``GPU_METRICS.metricId`` on modern exports; skip binding.
        return None
    tables = list_tables(conn)
    best: Optional[Tuple[int, Dict[str, Any]]] = None

    for st in tables:
        stl = st.lower()
        if "demangle" in stl or "symbol" in stl:
            continue
        if not _is_likely_string_dictionary_table(conn, st):
            continue
        for tc in _string_table_text_columns(conn, st):
            id_to_label = _collect_dram_bandwidth_string_ids(conn, st, tc)
            if len(id_to_label) < 1:
                continue
            ids = list(id_to_label.keys())

            for t in tables:
                if t == st:
                    continue
                tl = t.lower()
                if "stringid" in tl or "demangle" in tl:
                    continue
                tcols = _table_columns(conn, t)
                tcl = {c.lower(): c for c in tcols}
                ts_col = val_col = None
                for pattern, pts in (
                    ("timestamp", 3),
                    ("timestamps", 3),
                    ("start", 2),
                    ("time", 1),
                ):
                    for c, orig in tcl.items():
                        if pattern in c and "duration" not in c:
                            ts_col = orig
                            break
                    if ts_col:
                        break
                if not ts_col:
                    for key in ("ts", "tscode", "timens", "timestampsns", "starttime", "start_time"):
                        if key in tcl:
                            ts_col = tcl[key]
                            break
                for c, orig in tcl.items():
                    if c in ("value", "values", "metricvalue", "samplevalue"):
                        val_col = orig
                        break
                if not ts_col or not val_col:
                    continue

                mid_cols = _metric_reference_column_names(tcl)
                if not mid_cols and ("gpumetric" in tl or "gpu_metric" in tl) and "id" in tcl:
                    mid_cols = [tcl["id"]]
                for mid_c in mid_cols:
                    cnt = _count_rows_matching_metric_ids(conn, t, mid_c, ids)
                    if cnt < 1:
                        continue
                    bonus = 0
                    if "gpumetric" in tl or "gpu_metric" in tl:
                        bonus += 10_000_000
                    if "sample" in tl:
                        bonus += 100_000
                    score = cnt + bonus
                    cand = {
                        "table": t,
                        "score": 10_000 + score,
                        "timestamp_col": ts_col,
                        "value_col": val_col,
                        "metric_id_col": mid_c,
                        "name_col": None,
                        "gpu_col": next((tcl[k] for k in tcl if "gpu" in k and "id" in k), None),
                        "string_ids_table": st,
                    }
                    if best is None or score > best[0]:
                        best = (score, cand)

    return best[1] if best else None


def _count_rows_matching_metric_ids(
    conn: sqlite3.Connection, table: str, mid_col: str, ids: List[int]
) -> int:
    if not ids:
        return 0
    total = 0
    chunk_size = 400
    for off in range(0, len(ids), chunk_size):
        chunk = ids[off : off + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        for qfmt in (
            f'SELECT COUNT(*) AS c FROM "{table}" WHERE CAST("{mid_col}" AS INTEGER) IN ({placeholders})',
            f'SELECT COUNT(*) AS c FROM "{table}" WHERE "{mid_col}" IN ({placeholders})',
        ):
            try:
                row = conn.execute(qfmt, chunk).fetchone()
                if row is None:
                    break
                total += int(row[0])
                break
            except sqlite3.Error:
                continue
    return total


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


def _labels_look_like_nvtx_string_table(labels: Sequence[str]) -> bool:
    """True when ``metric_id`` resolved to NVTX / env blobs instead of GPU counter names."""
    if not labels:
        return True
    ncheck = min(80, len(labels))
    bad = 0
    for L in labels[:ncheck]:
        if len(L) > 500:
            bad += 1
            continue
        low = L.lower()
        if L.startswith("/") and ("nsight" in low or "/opt/" in low or "/bin/" in low):
            bad += 1
        elif "=" in L and len(L) > 200 and ("path=" in low or "conda_" in low or "microbench/" in low):
            bad += 1
    return bad >= max(2, ncheck // 5)


def _distinct_metric_labels_for_candidate(
    conn: sqlite3.Connection,
    cand: Dict[str, Any],
    string_ids_table: Optional[str],
    *,
    max_distinct: int = 384,
) -> List[str]:
    """Cheap distinct labels from one candidate table (inline name column and/or string-id resolution)."""
    t = cand["table"]
    name_c = cand.get("name_col")
    mid_c = cand.get("metric_id_col")
    sid_use = cand.get("string_ids_table") or string_ids_table
    out: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    try:
        if _is_gpu_metrics_sample_table(t):
            cat = _gpu_metric_name_catalog(conn)
            if cat:
                nc = cat["name_col"]
                ct = cat["table"]
                qc = f'SELECT DISTINCT "{nc}" AS v FROM "{ct}" WHERE "{nc}" IS NOT NULL LIMIT {max_distinct}'
                for r in conn.execute(qc):
                    add(str(r["v"] or ""))
                if len(out) >= 8:
                    return out
        if name_c:
            q = f'SELECT DISTINCT "{name_c}" AS v FROM "{t}" WHERE "{name_c}" IS NOT NULL LIMIT {max_distinct}'
            for r in conn.execute(q):
                add(str(r["v"] or ""))
        if mid_c and len(out) < 8:
            qm = f'SELECT DISTINCT "{mid_c}" AS mid FROM "{t}" WHERE "{mid_c}" IS NOT NULL LIMIT {max_distinct}'
            for r in conn.execute(qm):
                add(_resolve_string(conn, sid_use, r["mid"]))
    except sqlite3.Error:
        pass
    return out


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

    cat = _gpu_metric_name_catalog(conn)
    colmap = {c.lower(): c for c in _table_columns(conn, t)}
    type_c = colmap.get("typeid")
    use_catalog_join = bool(
        cat is not None
        and type_c is not None
        and mid_c is not None
        and _is_gpu_metrics_sample_table(t)
    )

    dram_metric_ids: Optional[List[int]] = None
    if use_catalog_join and cat is not None:
        dram_metric_ids = _catalog_dram_bandwidth_metric_ids(conn, cat)
        if dram_metric_ids:
            _nsys_hbm_status(
                f"GPU_METRICS SQL filter: {len(dram_metric_ids)} DRAM/HBM bandwidth series "
                f"(omitting other GPU counters — large speedup)"
            )
        else:
            _nsys_hbm_status(
                "GPU_METRICS: no DRAM bandwidth rows matched in catalog — loading all metric samples (slow)"
            )

    if use_catalog_join and cat is not None:
        ga, ca = "g", "c"
        ct = cat["table"]
        c_type, c_mid, c_name = cat["type_col"], cat["metric_id_col"], cat["name_col"]
        sel = [
            f'{ga}."{ts_c}" AS ts_raw',
            f'{ga}."{val_c}" AS val_raw',
            f'{ca}."{c_name}" AS name_inline',
            f'{ga}."{mid_c}" AS metric_id',
        ]
        if gpu_c:
            sel.append(f'{ga}."{gpu_c}" AS gpu_id')
        q = (
            f'SELECT {", ".join(sel)} FROM "{t}" AS {ga} '
            f'INNER JOIN "{ct}" AS {ca} '
            f'ON {ga}."{type_c}" = {ca}."{c_type}" AND {ga}."{mid_c}" = {ca}."{c_mid}"'
        )
        q_params: Tuple[Any, ...] = ()
        wh: List[str] = []
        if gpu_id is not None and gpu_c:
            wh.append(f'{ga}."{gpu_c}" = ?')
            q_params = (int(gpu_id),)
        if dram_metric_ids:
            ph = ",".join("?" * len(dram_metric_ids))
            wh.append(f'{ga}."{mid_c}" IN ({ph})')
            q_params = q_params + tuple(int(x) for x in dram_metric_ids)
        if wh:
            q += " WHERE " + " AND ".join(wh)
    else:
        sel = [f'"{ts_c}" AS ts_raw', f'"{val_c}" AS val_raw']
        if mid_c:
            sel.append(f'"{mid_c}" AS metric_id')
        if name_c:
            sel.append(f'"{name_c}" AS name_inline')
        if gpu_c:
            sel.append(f'"{gpu_c}" AS gpu_id')
        q = f'SELECT {", ".join(sel)} FROM "{t}"'
        q_params = ()
        if gpu_id is not None and gpu_c:
            q += f' WHERE "{gpu_c}" = ?'
            q_params = (int(gpu_id),)

    rows = []
    n_loaded = 0
    next_log = _NS_HBM_SAMPLE_FETCH_LOG_EVERY
    cur = conn.execute(q, q_params) if q_params else conn.execute(q)
    for r in cur:
        d = dict(r)
        tr, vr = d.get("ts_raw"), d.get("val_raw")
        if tr is None or vr is None:
            continue
        try:
            ts = int(float(tr))
            val = float(vr)
        except (TypeError, ValueError):
            continue
        mid = d.get("metric_id")
        label_inline = (d.get("name_inline") or "").strip()
        if use_catalog_join:
            resolved = ""
        else:
            resolved = _resolve_string(conn, sid_tab, mid).strip() if mid is not None else ""
        if _label_is_obviously_not_a_metric_name(label_inline):
            label = resolved or label_inline
        else:
            label = label_inline or resolved
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
        n_loaded += 1
        if n_loaded >= next_log:
            _nsys_hbm_status(f"GPU metric samples fetched: {n_loaded:,} rows (still reading SQLite…)…")
            next_log += _NS_HBM_SAMPLE_FETCH_LOG_EVERY
    if n_loaded > 0:
        _nsys_hbm_status(f"GPU metric samples: {n_loaded:,} rows fetched — sorting…")
    rows.sort(key=lambda x: (x["timestamp_ns"], x["metric_name"]))
    if n_loaded > 0:
        _nsys_hbm_status(f"GPU metric samples: sort done ({len(rows):,} rows)")
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
                sr, er = d.get("s_raw"), d.get("e_raw")
                if sr is None or er is None:
                    continue
                try:
                    s = int(float(sr))
                    e = int(float(er))
                except (TypeError, ValueError):
                    continue
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


def _min_chrome_aten_ts_us(trace: Dict[str, Any]) -> Optional[float]:
    """Earliest ``aten::`` cpu_op timestamp in a Chrome trace (µs)."""
    m: Optional[float] = None
    for e in trace.get("traceEvents") or []:
        if e.get("ph") != "X":
            continue
        if e.get("cat") != "cpu_op":
            continue
        n = e.get("name") or ""
        if not str(n).startswith("aten::"):
            continue
        try:
            t = float(e.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if m is None or t < m:
            m = t
    return m


def _load_string_ids_map(conn: sqlite3.Connection, tables: Optional[List[str]] = None) -> Dict[int, str]:
    """Map StringIds row id → text for resolving CUPTI nameId."""
    tables = tables or list_tables(conn)
    st = _find_string_ids_table(conn, tables)
    if not st:
        return {}
    cols = _table_columns(conn, st)
    id_col = None
    for c in cols:
        if c.lower() in ("id", "pk", "stringid"):
            id_col = c
            break
    if id_col is None:
        id_col = list(cols.keys())[0]
    tx_cols = _string_table_text_columns(conn, st)
    if not tx_cols:
        return {}
    tx_col = tx_cols[0]
    out: Dict[int, str] = {}
    try:
        for r in conn.execute(f'SELECT "{id_col}", "{tx_col}" FROM "{st}"'):
            try:
                out[int(r[0])] = str(r[1])
            except (TypeError, ValueError):
                continue
    except sqlite3.Error:
        return {}
    return out


def load_cuda_runtime_launch_events(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Load CUDA host-side launch API rows from Nsight SQLite (CUPTI runtime table).

    Used to synthesize Chrome ``cuda_runtime`` / ``cuda_driver`` events so traces
    captured under ``nsys`` (CPU-only PyTorch profiler) still expose
    ``launches_by_corr`` for linking.
    """
    tables = list_tables(conn)
    rt_table = None
    for t in tables:
        if t.upper() == "CUPTI_ACTIVITY_KIND_RUNTIME":
            rt_table = t
            break
    if not rt_table:
        for t in tables:
            u = t.upper()
            if "CUPTI" in u and "RUNTIME" in u and "CALLBACK" not in u and "KERNEL" not in u:
                rt_table = t
                break
    if not rt_table:
        return []

    cols = _table_columns(conn, rt_table)
    cl = {c.lower(): c for c in cols}
    start_c = cl.get("start") or cl.get("globalstart") or cl.get("global_start")
    end_c = cl.get("end") or cl.get("globalend") or cl.get("global_end")
    corr_c = cl.get("correlationid") or cl.get("correlation")
    name_id_c = cl.get("nameid")
    if not start_c or not end_c or not corr_c:
        return []

    strmap = _load_string_ids_map(conn, tables) if name_id_c else {}
    out: List[Dict[str, Any]] = []
    sel_cols = [start_c, end_c, corr_c]
    if name_id_c:
        sel_cols.append(name_id_c)
    q = ", ".join(f'"{c}"' for c in sel_cols)
    try:
        for r in conn.execute(f'SELECT {q} FROM "{rt_table}"'):
            row = {c: r[c] for c in sel_cols}
            try:
                s_ns = int(float(row[start_c]))
                e_ns = int(float(row[end_c]))
            except (TypeError, ValueError):
                continue
            if e_ns <= s_ns:
                continue
            try:
                corr_i = int(row[corr_c]) if row.get(corr_c) is not None else None
            except (TypeError, ValueError):
                corr_i = None
            if corr_i is None:
                continue
            nm = ""
            if name_id_c and row.get(name_id_c) is not None:
                try:
                    nid = int(row[name_id_c])
                    nm = strmap.get(nid, "")
                except (TypeError, ValueError):
                    nm = ""
            if not nm or ("LaunchKernel" not in nm and "GraphLaunch" not in nm):
                continue
            out.append(
                {
                    "start_ns": s_ns,
                    "end_ns": e_ns,
                    "name": nm,
                    "correlation": corr_i,
                }
            )
    except sqlite3.Error as ex:
        LOGGER.warning("load_cuda_runtime_launch_events: skipping table %s: %s", rt_table, ex)
        return []

    out.sort(key=lambda x: (x["start_ns"], x["end_ns"]))
    return out


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

    # HBM / device memory (Nsight GB20x and newer often omit the word "DRAM").
    bw_tokens = ("bandwidth", "throughput", "bw", "%")
    if read_pick is None:
        for m, low in lower_map.items():
            if "read" not in low:
                continue
            if ("hbm" in low or "dram" in low) and any(t in low for t in bw_tokens):
                read_pick = m
                semantic[m] = "dram_read_bw"
                break
    if write_pick is None:
        for m, low in lower_map.items():
            if "write" not in low:
                continue
            if ("hbm" in low or "dram" in low) and any(t in low for t in bw_tokens):
                write_pick = m
                semantic[m] = "dram_write_bw"
                break
    if read_pick is None:
        for m, low in lower_map.items():
            if "device" in low and "memory" in low and "read" in low and any(
                t in low for t in bw_tokens
            ):
                read_pick = m
                semantic[m] = "dram_read_bw"
                break
    if write_pick is None:
        for m, low in lower_map.items():
            if "device" in low and "memory" in low and "write" in low and any(
                t in low for t in bw_tokens
            ):
                write_pick = m
                semantic[m] = "dram_write_bw"
                break

    # "GPU Memory Read/Write Throughput" (GB20x / Nsight 2025.x; labels omit "device"/"DRAM").
    if read_pick is None:
        for m, low in lower_map.items():
            if "read" not in low or "gpu" not in low or "memory" not in low:
                continue
            if any(x in low for x in ("host", "pageable", "pinned", "sysmem")):
                continue
            if "system" in low and "memory" in low:
                continue
            if any(t in low for t in bw_tokens):
                read_pick = m
                semantic[m] = "dram_read_bw"
                break
    if write_pick is None:
        for m, low in lower_map.items():
            if "write" not in low or "gpu" not in low or "memory" not in low:
                continue
            if any(x in low for x in ("host", "pageable", "pinned", "sysmem")):
                continue
            if "system" in low and "memory" in low:
                continue
            if any(t in low for t in bw_tokens):
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


# Process-pool workers for ``samples_to_byte_windows`` (large time unions).
_STBW_TU: List[int] = []
_STBW_RS: List[Tuple[int, float]] = []
_STBW_WS: List[Tuple[int, float]] = []
_STBW_PEAK: Optional[float] = None
_STBW_KIND: str = "percent"


def _stbw_worker_init(
    ts_union: List[int],
    read_series: List[Tuple[int, float]],
    write_series: List[Tuple[int, float]],
    peak_bps: Optional[float],
    kind: str,
) -> None:
    global _STBW_TU, _STBW_RS, _STBW_WS, _STBW_PEAK, _STBW_KIND
    _STBW_TU = ts_union
    _STBW_RS = read_series
    _STBW_WS = write_series
    _STBW_PEAK = peak_bps
    _STBW_KIND = kind


def _stbw_worker_to_bps(val: float) -> float:
    k = _STBW_KIND
    pb = _STBW_PEAK
    if k == "percent" and pb is not None:
        return max(0.0, val) / 100.0 * pb
    if k == "bytes_per_sec":
        return max(0.0, val)
    raise RuntimeError(f"Unsupported metric kind for conversion: {k}")


def _stbw_worker_build_slice(irange: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Build window dicts for ``ts_union[i0:i1+1]`` segments (indices into ``ts_union``)."""
    i0, i1 = irange
    tu = _STBW_TU
    rs, ws = _STBW_RS, _STBW_WS
    out: List[Dict[str, Any]] = []
    for i in range(i0, i1):
        t0, t1 = tu[i], tu[i + 1]
        if t1 <= t0:
            continue
        rv = _stbw_worker_to_bps(_series_last_value_at_or_before(rs, t0))
        wv = _stbw_worker_to_bps(_series_last_value_at_or_before(ws, t0))
        dt_s = (t1 - t0) * 1e-9
        rb = rv * dt_s
        wb = wv * dt_s
        out.append(
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
    return out


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

    read_series = samples_by_metric.get(read_metric or "", [])
    write_series = samples_by_metric.get(write_metric or "", [])
    # Only union timestamps from the read/write series actually used (not every column in the table).
    ts_set: Set[int] = set()
    for t, _ in read_series:
        ts_set.add(int(t))
    for t, _ in write_series:
        ts_set.add(int(t))
    ts_union: List[int] = sorted(ts_set)
    if len(ts_union) < 2 and last_window_ns is None:
        raise RuntimeError("Not enough GPU metric samples to form windows.")

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
    n_inner = len(ts_union) - 1
    workers = _nsys_hbm_parallel_workers()
    if n_inner >= 12000 and workers > 1:
        n_chunks = min(workers, max(1, n_inner // 6000))
        step = (n_inner + n_chunks - 1) // n_chunks
        ranges: List[Tuple[int, int]] = []
        lo = 0
        while lo < n_inner:
            hi = min(lo + step, n_inner)
            ranges.append((lo, hi))
            lo = hi
        _nsys_hbm_status(
            f"Building {n_inner:,} DRAM byte windows ({len(ranges)} parallel chunks, "
            f"≤{workers} workers)…"
        )
        with ProcessPoolExecutor(
            max_workers=len(ranges),
            initializer=_stbw_worker_init,
            initargs=(ts_union, read_series, write_series, peak_bps, kind),
        ) as ex:
            parts = list(ex.map(_stbw_worker_build_slice, ranges))
        for part in parts:
            windows.extend(part)
    else:
        for i in range(n_inner):
            t0, t1 = ts_union[i], ts_union[i + 1]
            if t1 <= t0:
                continue
            rv = to_bps(_series_last_value_at_or_before(read_series, t0))
            wv = to_bps(_series_last_value_at_or_before(write_series, t0))
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
        rv = to_bps(_series_last_value_at_or_before(read_series, t_last))
        wv = to_bps(_series_last_value_at_or_before(write_series, t_last))
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


_AW_INTV_SPANS: List[Tuple[int, int]] = []
_AW_ICS: List[int] = []
_AW_ICL: List[List[int]] = []


def _aw_pool_init(
    intv_spans: List[Tuple[int, int]],
    ichunk_starts: List[int],
    ichunk_lists: List[List[int]],
) -> None:
    global _AW_INTV_SPANS, _AW_ICS, _AW_ICL
    _AW_INTV_SPANS = intv_spans
    _AW_ICS = ichunk_starts
    _AW_ICL = ichunk_lists


def _aw_pool_process_chunk(
    windows_chunk: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n = len(_AW_INTV_SPANS)
    acc_read = np.zeros(n, dtype=np.float64)
    acc_write = np.zeros(n, dtype=np.float64)
    acc_tot = np.zeros(n, dtype=np.float64)
    ur = uw = ut = 0.0
    for w in windows_chunk:
        ws = int(w["start_ns"])
        we = int(w["end_ns"])
        rb = float(w["read_bytes"])
        wb = float(w["write_bytes"])
        tb = float(w["total_bytes"])
        overlaps: List[Tuple[int, int]] = []
        for idx in _span_indices_touching_interval(
            ws, we, _AW_ICS, _AW_ICL, _NS_OVERLAP_CHUNK_NS
        ):
            ist, ien = _AW_INTV_SPANS[idx]
            ov = compute_overlap_ns(ws, we, ist, ien)
            if ov > 0:
                overlaps.append((idx, ov))
        denom = sum(o for _, o in overlaps)
        if denom <= 0:
            ur += rb
            uw += wb
            ut += tb
            continue
        for idx, raw_ov in overlaps:
            sh = raw_ov / denom
            acc_read[idx] += rb * sh
            acc_write[idx] += wb * sh
            acc_tot[idx] += tb * sh
    return acc_read, acc_write, acc_tot, ur, uw, ut


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
    intv_spans: List[Tuple[int, int]] = []
    for it in intervals:
        intv_spans.append(
            (int(it.get("start_ns", 0) or 0), int(it.get("end_ns", 0) or 0))
        )
    ichunk_starts, ichunk_lists = _build_ns_chunk_index(
        intv_spans,
        _NS_OVERLAP_CHUNK_NS,
        progress_label="GPU intervals → timeline index",
    )

    nw = len(windows)
    workers = _nsys_hbm_parallel_workers()
    use_parallel = nw >= 8000 and workers > 1 and n > 0

    if use_parallel:
        n_chunks = min(workers, max(1, nw // 4000))
        step = (nw + n_chunks - 1) // n_chunks
        chunks: List[List[Dict[str, Any]]] = []
        lo = 0
        wlist = list(windows)
        while lo < nw:
            hi = min(lo + step, nw)
            chunks.append(wlist[lo:hi])
            lo = hi
        _nsys_hbm_status(
            f"sample windows → interval HBM — {nw:,} windows in {len(chunks)} parallel chunks "
            f"(≤{workers} workers)…"
        )
        with ProcessPoolExecutor(
            max_workers=len(chunks),
            initializer=_aw_pool_init,
            initargs=(intv_spans, ichunk_starts, ichunk_lists),
        ) as ex:
            parts = list(ex.map(_aw_pool_process_chunk, chunks))
        acc_read = np.zeros(n, dtype=np.float64)
        acc_write = np.zeros(n, dtype=np.float64)
        acc_tot = np.zeros(n, dtype=np.float64)
        ur = uw = ut = 0.0
        for p in parts:
            acc_read += p[0]
            acc_write += p[1]
            acc_tot += p[2]
            ur += p[3]
            uw += p[4]
            ut += p[5]
        acc_read_l = acc_read.tolist()
        acc_write_l = acc_write.tolist()
        acc_tot_l = acc_tot.tolist()
    else:
        acc_read_l = [0.0] * n
        acc_write_l = [0.0] * n
        acc_tot_l = [0.0] * n
        ur = uw = ut = 0.0

        win_prog = _NsysHbmMilestones("sample windows → interval HBM", nw)
        for wi, w in enumerate(windows):
            ws = int(w["start_ns"])
            we = int(w["end_ns"])
            rb = float(w["read_bytes"])
            wb = float(w["write_bytes"])
            tb = float(w["total_bytes"])

            overlaps: List[Tuple[int, int]] = []
            for idx in _span_indices_touching_interval(
                ws, we, ichunk_starts, ichunk_lists, _NS_OVERLAP_CHUNK_NS
            ):
                ist, ien = intv_spans[idx]
                ov = compute_overlap_ns(ws, we, ist, ien)
                if ov > 0:
                    overlaps.append((idx, ov))
            denom = sum(o for _, o in overlaps)
            if denom <= 0:
                ur += rb
                uw += wb
                ut += tb
                continue
            for idx, raw_ov in overlaps:
                sh = raw_ov / denom
                acc_read_l[idx] += rb * sh
                acc_write_l[idx] += wb * sh
                acc_tot_l[idx] += tb * sh
            win_prog.step(wi + 1)

    out = []
    for i, it in enumerate(intervals):
        out.append(
            {
                **it,
                "nsys_hbm_estimated_read_bytes": acc_read_l[i],
                "nsys_hbm_estimated_write_bytes": acc_write_l[i],
                "nsys_hbm_estimated_total_bytes": acc_tot_l[i],
                "nsys_hbm_attribution_method": "sample_overlap",
                "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
            }
        )
    return out, ur, uw, ut


def _soda_gpu_events_from_trace(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Same ordering as MoE ``_iter_gpu_events_ordered`` (kernels + mem ops)."""
    out: List[Dict[str, Any]] = []
    evs = trace.get("traceEvents") or []
    n_e = len(evs)
    _sg_next = 500_000
    for ei, event in enumerate(evs):
        if n_e >= 500_000 and ei >= _sg_next:
            _nsys_hbm_status(f"Scanning Chrome events for GPU rows: {ei:,}/{n_e:,}…")
            _sg_next += 500_000
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

    Nsight exports GPU intervals on a **capture-relative** clock while PyTorch
    ``aten::`` events use the profiler's **absolute** µs timeline. Without a shift,
    every kernel looks like it runs *before* any ATen op and correlation-based
    linking fails. We align injected GPU + CUDA launch rows to the earliest
    ``aten::`` timestamp in the trace (see ``nsys_ts_align_delta_us``).

    Also appends matching CUDA host launch rows from ``CUPTI_ACTIVITY_KIND_RUNTIME``
    so ``cudaLaunchKernel`` / ``cuLaunchKernel`` appear in the Chrome trace and
    ``launches_by_corr`` is populated.

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
        launch_rows = load_cuda_runtime_launch_events(conn)
    finally:
        conn.close()

    new_events: List[Dict[str, Any]] = []
    inj_prog = _NsysHbmMilestones(
        "SQLite GPU intervals → Chrome trace events",
        len(intervals),
        min_total=_NS_HBM_INDEX_PROGRESS_MIN,
    )
    for ii, it in enumerate(intervals):
        s_ns = int(it["start_ns"])
        e_ns = int(it["end_ns"])
        if e_ns > s_ns:
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
        inj_prog.step(ii + 1)

    launch_events: List[Dict[str, Any]] = []
    for lr in launch_rows:
        s_ns = int(lr["start_ns"])
        e_ns = int(lr["end_ns"])
        if e_ns <= s_ns:
            continue
        nm = str(lr.get("name") or "cudaLaunchKernel")
        lcat = "cuda_driver" if nm.startswith("cu") else "cuda_runtime"
        c = lr.get("correlation")
        launch_events.append(
            {
                "ph": "X",
                "cat": lcat,
                "name": nm,
                "pid": 0,
                "tid": 0,
                "ts": s_ns / 1000.0,
                "dur": (e_ns - s_ns) / 1000.0,
                "args": {
                    "nsys_injected": True,
                    "correlation": c,
                },
            }
        )

    combined = new_events + launch_events
    aten_min = _min_chrome_aten_ts_us(trace)
    inj_min_ts: Optional[float] = None
    for ev in combined:
        try:
            t = float(ev.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        if inj_min_ts is None or t < inj_min_ts:
            inj_min_ts = t
    delta_us: Optional[float] = None
    if aten_min is not None and inj_min_ts is not None and combined:
        delta_us = aten_min - inj_min_ts
        trace["nsys_ts_align_delta_us"] = delta_us
        for ev in combined:
            ev["ts"] = float(ev["ts"]) + delta_us
    else:
        trace.setdefault("nsys_ts_align_delta_us", None)

    trace.setdefault("traceEvents", []).extend(combined)
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)

    return len(combined)


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
    if diffs:
        diffs.sort()
        return diffs[len(diffs) // 2]
    # No overlapping correlation IDs: PyTorch Chrome ``ts`` (µs) and Nsight
    # ``start_ns`` are usually different clock domains. Align earliest Nsight
    # interval start to earliest trace GPU span so sample_overlap can find overlap.
    if not nsys_intervals or not soda_events:
        return 0
    min_nsys = min(int(it["start_ns"]) for it in nsys_intervals)
    min_soda = min(int(float(ev["ts_us"]) * US_TO_NS) for ev in soda_events)
    off = min_nsys - min_soda
    LOGGER.info(
        "nsys-hbm: time offset from min-start alignment (no correlation pairs): %s ns",
        off,
    )
    _nsys_hbm_status(
        "GPU↔Nsight timeline: using earliest-interval alignment "
        f"(no correlation overlap; offset_ns={off:,})"
    )
    return off


def _soda_trace_spans_index(
    soda_events: Sequence[Dict[str, Any]],
    offset_ns: int,
    *,
    progress_label: Optional[str] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[List[int]]]:
    """(ss, se) per trace GPU event in ns + chunk index for overlap queries."""
    spans: List[Tuple[int, int]] = []
    for ev in soda_events:
        ss = int(float(ev["ts_us"]) * US_TO_NS) + offset_ns
        se = int(float(ev["ts_us"] + ev["dur_us"]) * US_TO_NS) + offset_ns
        spans.append((ss, se))
    cs, cl = _build_ns_chunk_index(
        spans, _NS_OVERLAP_CHUNK_NS, progress_label=progress_label
    )
    return spans, cs, cl


_DIST_SODA_SPANS: List[Tuple[int, int]] = []
_DIST_CS: List[int] = []
_DIST_CL: List[List[int]] = []


def _dist_pool_init(
    soda_spans: List[Tuple[int, int]],
    chunk_starts: List[int],
    chunk_lists: List[List[int]],
) -> None:
    global _DIST_SODA_SPANS, _DIST_CS, _DIST_CL
    _DIST_SODA_SPANS = soda_spans
    _DIST_CS = chunk_starts
    _DIST_CL = chunk_lists


def _dist_pool_process_chunk(
    interval_chunk: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    n = len(_DIST_SODA_SPANS)
    acc_read = np.zeros(n, dtype=np.float64)
    acc_write = np.zeros(n, dtype=np.float64)
    acc_tot = np.zeros(n, dtype=np.float64)
    ur = uw = ut = 0.0
    for it in interval_chunk:
        is_ = int(it["start_ns"])
        ie_ = int(it["end_ns"])
        rb = float(it.get("nsys_hbm_estimated_read_bytes", 0) or 0)
        wb = float(it.get("nsys_hbm_estimated_write_bytes", 0) or 0)
        tb = float(it.get("nsys_hbm_estimated_total_bytes", 0) or 0)
        overlaps: List[Tuple[int, int]] = []
        for j in _span_indices_touching_interval(
            is_, ie_, _DIST_CS, _DIST_CL, _NS_OVERLAP_CHUNK_NS
        ):
            ss, se = _DIST_SODA_SPANS[j]
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
    return acc_read, acc_write, acc_tot, ur, uw, ut


def distribute_interval_bytes_to_soda_events(
    interval_attrs: Sequence[Dict[str, Any]],
    soda_events: Sequence[Dict[str, Any]],
) -> Tuple[
    Dict[int, Dict[str, Any]],
    float,
    float,
    float,
    int,
    List[Tuple[int, int]],
    List[int],
    List[List[int]],
]:
    """
    For each Nsight interval, split its attributed bytes across overlapping SODA GPU
    events proportionally to overlap duration (same rule as sample windows).

    Returns:
        (soda_index -> field dict, trace_unassigned_read, trace_unassigned_write,
         trace_unassigned_total, time_offset_ns, soda_spans_ns, chunk_starts, chunk_lists)
        The last three entries repeat the trace timeline index used for overlap queries
        (``soda_spans_ns`` aligns with ``soda_events`` indices).
    """
    offset_ns = _estimate_time_offset_ns(soda_events, interval_attrs)
    n = len(soda_events)

    soda_spans, chunk_starts, chunk_lists = _soda_trace_spans_index(
        soda_events,
        offset_ns,
        progress_label="trace GPU events → timeline index",
    )

    ni = len(interval_attrs)
    workers = _nsys_hbm_parallel_workers()
    use_parallel = ni >= 8000 and workers > 1 and n > 0

    if use_parallel:
        n_chunks = min(workers, max(1, ni // 4000))
        step = (ni + n_chunks - 1) // n_chunks
        ichunks: List[List[Dict[str, Any]]] = []
        lo = 0
        ial = list(interval_attrs)
        while lo < ni:
            hi = min(lo + step, ni)
            ichunks.append(ial[lo:hi])
            lo = hi
        _nsys_hbm_status(
            f"Nsight intervals → trace kernel bytes — {ni:,} intervals in {len(ichunks)} "
            f"parallel chunks (≤{workers} workers)…"
        )
        with ProcessPoolExecutor(
            max_workers=len(ichunks),
            initializer=_dist_pool_init,
            initargs=(soda_spans, chunk_starts, chunk_lists),
        ) as ex:
            parts = list(ex.map(_dist_pool_process_chunk, ichunks))
        acc_read = np.zeros(n, dtype=np.float64)
        acc_write = np.zeros(n, dtype=np.float64)
        acc_tot = np.zeros(n, dtype=np.float64)
        ur = uw = ut = 0.0
        for p in parts:
            acc_read += p[0]
            acc_write += p[1]
            acc_tot += p[2]
            ur += p[3]
            uw += p[4]
            ut += p[5]
        acc_read_l = acc_read.tolist()
        acc_write_l = acc_write.tolist()
        acc_tot_l = acc_tot.tolist()
    else:
        acc_read_l = [0.0] * n
        acc_write_l = [0.0] * n
        acc_tot_l = [0.0] * n
        ur = uw = ut = 0.0

        dist_prog = _NsysHbmMilestones("Nsight intervals → trace kernel bytes", ni)
        for ki, it in enumerate(interval_attrs):
            is_ = int(it["start_ns"])
            ie_ = int(it["end_ns"])
            rb = float(it.get("nsys_hbm_estimated_read_bytes", 0) or 0)
            wb = float(it.get("nsys_hbm_estimated_write_bytes", 0) or 0)
            tb = float(it.get("nsys_hbm_estimated_total_bytes", 0) or 0)
            overlaps: List[Tuple[int, int]] = []
            for j in _span_indices_touching_interval(
                is_, ie_, chunk_starts, chunk_lists, _NS_OVERLAP_CHUNK_NS
            ):
                ss, se = soda_spans[j]
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
                acc_read_l[j] += rb * sh
                acc_write_l[j] += wb * sh
                acc_tot_l[j] += tb * sh
            dist_prog.step(ki + 1)

    out: Dict[int, Dict[str, Any]] = {}
    for j in range(n):
        matched = acc_tot_l[j] > 0
        out[j] = {
            "nsys_hbm_estimated_read_bytes": acc_read_l[j] if matched else None,
            "nsys_hbm_estimated_write_bytes": acc_write_l[j] if matched else None,
            "nsys_hbm_estimated_total_bytes": acc_tot_l[j] if matched else None,
            "nsys_hbm_attribution_method": "sample_overlap",
            "nsys_hbm_metric_source": "nsight_systems_gpu_metrics",
            "nsys_hbm_match_status": "matched" if matched else "unmatched",
            "nsys_time_offset_ns_applied": offset_ns,
        }
    return out, ur, uw, ut, offset_ns, soda_spans, chunk_starts, chunk_lists


_UM_SPANS: List[Tuple[int, int]] = []
_UM_CS: List[int] = []
_UM_CL: List[List[int]] = []


def _unmatched_trace_overlap_init(
    spans_um: List[Tuple[int, int]],
    cs_um: List[int],
    cl_um: List[List[int]],
) -> None:
    global _UM_SPANS, _UM_CS, _UM_CL
    _UM_SPANS = spans_um
    _UM_CS = cs_um
    _UM_CL = cl_um


def _unmatched_trace_overlap_chunk(
    interval_chunk: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Intervals with zero overlap against the SODA trace GPU span index."""
    out: List[Dict[str, Any]] = []
    for it in interval_chunk:
        is_ = int(it.get("start_ns", 0) or 0)
        ie_ = int(it.get("end_ns", 0) or 0)
        hit = False
        for j in _span_indices_touching_interval(
            is_, ie_, _UM_CS, _UM_CL, _NS_OVERLAP_CHUNK_NS
        ):
            ss, se = _UM_SPANS[j]
            if compute_overlap_ns(is_, ie_, ss, se) > 0:
                hit = True
                break
        if not hit:
            out.append({k: v for k, v in it.items() if not k.startswith("nsys_hbm_")})
    return out


def patch_trace_json_with_nsys(
    trace_path: Path,
    soda_match: Dict[int, Dict[str, Any]],
) -> None:
    """Write nsys fields into matching trace event ``args`` (in file order)."""
    nbytes = trace_path.stat().st_size
    _nsys_hbm_status(
        f"Reading {trace_path.name} for HBM patch ({nbytes / (1024**2):.1f} MiB on disk)…"
    )
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    n_all = len(trace.get("traceEvents") or [])
    _nsys_hbm_status(
        f"Patching nsys_hbm_* onto GPU kernel/mem events (scanning {n_all:,} Chrome events)…"
    )
    idx = 0
    _next_gp = 200_000
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        m = soda_match.get(idx)
        idx += 1
        if idx >= _next_gp:
            _nsys_hbm_status(f"GPU row patch progress: {idx:,} kernel/mem events processed…")
            _next_gp += 200_000
        if not m:
            continue
        args = ev.setdefault("args", {})
        for k, v in m.items():
            if k not in args:
                args[k] = v

    _nsys_hbm_status(
        f"Writing patched {trace_path.name} ({n_all:,} events — large JSON may take minutes)…"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f)
    _nsys_hbm_status(f"Finished writing patched trace ({trace_path.name}).")


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

    _nsys_hbm_status("Nsight HBM: reading SQLite (DRAM metrics + GPU execution intervals)…")

    conn = open_sqlite(sqlite_path)
    try:
        _nsys_hbm_status("Scanning SQLite schema for GPU metric / CUDA tables…")
        mschema = discover_gpu_metrics_schema(conn)
        n_cand = len(mschema.get("metric_candidates") or [])
        _nsys_hbm_status(f"Schema scan done — {n_cand} GPU metric table candidate(s)")
        binding = find_gpu_dram_metric_sample_binding(conn)
        if binding:
            LOGGER.info(
                "Located GPU DRAM/HBM metric samples via string-id binding: table=%s column=%s strings=%s",
                binding["table"],
                binding.get("metric_id_col"),
                binding.get("string_ids_table"),
            )
            mc0 = mschema.get("metric_candidates") or []
            mschema = {**mschema, "metric_candidates": [binding] + mc0}
        sid_tab = mschema.get("string_ids_table")
        all_mc = mschema.get("metric_candidates") or []
        chosen: Optional[Dict[str, Any]] = None
        for cand in all_mc[:40]:
            labels = _distinct_metric_labels_for_candidate(conn, cand, sid_tab)
            if _labels_look_like_nvtx_string_table(labels):
                continue
            r_try, w_try, _ = classify_dram_bandwidth_metrics(labels)
            if r_try or w_try:
                chosen = cand
                break
        if chosen is None:
            for cand in all_mc[:40]:
                labels = _distinct_metric_labels_for_candidate(conn, cand, sid_tab)
                r_try, w_try, _ = classify_dram_bandwidth_metrics(labels)
                if r_try or w_try:
                    chosen = cand
                    break

        ordered_cands: List[Dict[str, Any]] = []
        seen_tables: set[str] = set()
        if chosen is not None:
            ordered_cands.append(chosen)
            seen_tables.add(str(chosen["table"]))
        for cand in all_mc[:40]:
            tkey = str(cand["table"])
            if tkey not in seen_tables:
                ordered_cands.append(cand)
                seen_tables.add(tkey)

        samples: List[Dict[str, Any]] = []
        by_name: Dict[str, List[Tuple[int, float]]] = {}
        read_m: Optional[str] = None
        write_m: Optional[str] = None
        names: List[str] = []
        _nsys_hbm_status(
            f"Loading DRAM/HBM metric time series (trying up to {len(ordered_cands)} table candidate(s))…"
        )
        for ci, cand in enumerate(ordered_cands):
            _nsys_hbm_status(f"Metric candidate {ci + 1}/{len(ordered_cands)}: table {cand['table']!r}")
            m_try = {**mschema, "metric_candidates": [cand]}
            samples = load_gpu_metric_samples(conn, gpu_id=gpu_id, schema_hint=m_try)
            by_name = {}
            for s in samples:
                by_name.setdefault(s["metric_name"], []).append((s["timestamp_ns"], s["value"]))
            for k in by_name:
                by_name[k].sort(key=lambda x: x[0])
            names = list(by_name.keys())
            plausible = [n for n in names if not _label_is_obviously_not_a_metric_name(n)]
            read_m, write_m, _sem = classify_dram_bandwidth_metrics(
                plausible if plausible else names
            )
            if read_m or write_m:
                break

        if read_metric_override:
            read_m = read_metric_override
        if write_metric_override:
            write_m = write_metric_override

        if not read_m and not write_m:
            conn.close()
            raise RuntimeError(
                "No DRAM/HBM read/write bandwidth metrics found in Nsight SQLite. "
                "Ensure capture used --gpu-metrics-devices and a metrics set that includes "
                "device DRAM or HBM bandwidth (see Nsight GPU Metrics for your chip). "
                f"Distinct metric names tried (up to 40): {names[:40]!r}"
            )

        _nsys_hbm_status(
            f"Classifying DRAM metrics and integrating {len(samples):,} samples into byte windows…"
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
        _nsys_hbm_status(
            f"SQLite phase done — {len(windows):,} bandwidth windows, {len(intervals):,} GPU intervals"
        )
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

    tp = Path(trace_path)
    tbytes = tp.stat().st_size
    _nsys_hbm_status(
        f"Loading {tp.name} for GPU↔interval HBM mapping "
        f"({tbytes / (1024**3):.2f} GiB on disk) — parsing JSON…"
    )
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    ntev = len(trace.get("traceEvents") or [])
    _nsys_hbm_status(f"Trace JSON parsed: {ntev:,} Chrome events — collecting GPU kernel/memcpy rows…")
    soda_ev = _soda_gpu_events_from_trace(trace)
    _nsys_hbm_status(f"Trace GPU rows collected: {len(soda_ev):,} (kernels + memcpy/memset) for byte mapping")
    soda_match, ur_tr, uw_tr, ut_tr, _off, _spans_um, _cs_um, _cl_um = (
        distribute_interval_bytes_to_soda_events(interval_attrs, soda_ev)
    )
    ni_um = len(interval_attrs)
    workers_um = _nsys_hbm_parallel_workers()
    if ni_um >= 8000 and workers_um > 1:
        n_chunks = min(workers_um, max(1, ni_um // 4000))
        step = (ni_um + n_chunks - 1) // n_chunks
        um_chunks: List[List[Dict[str, Any]]] = []
        lo = 0
        ial_um = list(interval_attrs)
        while lo < ni_um:
            hi = min(lo + step, ni_um)
            um_chunks.append(ial_um[lo:hi])
            lo = hi
        _nsys_hbm_status(
            f"Finding intervals with no trace overlap — {ni_um:,} intervals in {len(um_chunks)} "
            f"parallel chunks (≤{workers_um} workers)…"
        )
        with ProcessPoolExecutor(
            max_workers=len(um_chunks),
            initializer=_unmatched_trace_overlap_init,
            initargs=(_spans_um, _cs_um, _cl_um),
        ) as ex:
            um_parts = list(ex.map(_unmatched_trace_overlap_chunk, um_chunks))
        unmatched_rows = [row for part in um_parts for row in part]
    else:
        unmatched_rows = []
        um_prog = _NsysHbmMilestones("Finding intervals with no trace overlap", ni_um)
        for ui, it in enumerate(interval_attrs):
            is_ = int(it["start_ns"])
            ie_ = int(it["end_ns"])
            hit = False
            for j in _span_indices_touching_interval(
                is_, ie_, _cs_um, _cl_um, _NS_OVERLAP_CHUNK_NS
            ):
                ss, se = _spans_um[j]
                if compute_overlap_ns(is_, ie_, ss, se) > 0:
                    hit = True
                    break
            if not hit:
                unmatched_rows.append(
                    {k: v for k, v in it.items() if not k.startswith("nsys_hbm_")}
                )
            um_prog.step(ui + 1)
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

    _nsys_hbm_status(f"Nsight HBM attribution complete — artifacts under {out_dir}")

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
