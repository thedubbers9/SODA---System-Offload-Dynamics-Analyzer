# Abstraction: **execution-ish ordering** of kernel-database templates.
#
# Differs from op_profile: we never sort by op name here.  Order is the substrate
# for adjacency-based scratchpad modeling; losing it collapses distinct
# producer-consumer windows into meaningless aggregates.
#
# **Experimental:** trace timestamps are a best-effort proxy; async streams can
# still reorder real GPU execution.
"""Order classified kernel DB entries using ``trace.json`` when available.

Scratchpad reuse is about **adjacency and short lifetimes** in the dynamic
schedule.  Therefore **order fidelity** matters more than aggregating many
occurrences under one op label: we keep one list element per kernel-database
**template** in sorted order, not merged by ATen name.

We intentionally avoid collapsing “duplicate” kernel names here because the
kernel DB already keys on (op, shapes, kernel); two rows with the same ATen name
are still distinct templates and may appear at different execution indices after
trace ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from soda.common.data import clean_kernel_name


def trace_first_ts_per_kernel_name(trace_path: Path) -> Dict[str, float]:
    """Map cleaned GPU kernel name → first Chrome-trace ``ts`` (microseconds).

    **Architectural assumption:** the first trace appearance of a kernel template
    correlates better with program-side proximity than ``kernel_database.json``
    list order (often sorted by aggregate duration).  This remains approximate
    when multiple CUDA streams overlap.
    """
    try:
        data = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    events = data.get("traceEvents") or []
    first: Dict[str, float] = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        if ev.get("cat") != "kernel":
            continue
        name = clean_kernel_name(ev.get("name", "") or "")
        if not name:
            continue
        ts = float(ev.get("ts", 0.0))
        if name not in first or ts < first[name]:
            first[name] = ts
    return first


def order_classified_entries(
    classified_kernels: List[dict],
    trace_path: Optional[Path],
) -> Tuple[List[dict], str, str]:
    """Return ``(ordered_entries, ordering_source, warning_or_empty)``.

    **Primary:** sort by first GPU kernel timestamp per cleaned kernel name, then
    ``rank``, then entry id.

    **Fallback:** preserve kernel DB list order and emit an explicit warning that
    duration-based DB ordering is a weak proxy for program order.
    """
    if not classified_kernels:
        return [], "empty", ""

    if trace_path is not None and Path(trace_path).is_file():
        first_ts = trace_first_ts_per_kernel_name(Path(trace_path))

        def sort_key(e: dict) -> Tuple[float, int, str]:
            kn = (e.get("kernel") or {}).get("name", "") or ""
            ts = first_ts.get(kn, float("inf"))
            rank = int(e.get("rank", 10**9))
            eid = str(e.get("id", ""))
            return (ts, rank, eid)

        ordered = sorted(classified_kernels, key=sort_key)
        note = (
            "Ordering uses first GPU kernel timestamp per cleaned kernel name from "
            "trace.json; async streams may still reorder real execution."
        )
        return ordered, "trace_first_kernel_ts", note

    ordered = list(classified_kernels)
    warn = (
        "WARNING: No trace.json — experimental pipeline uses kernel_database.json list order, "
        "which is typically sorted by aggregate duration (not program order).  "
        "Treat groups, stage roles, and buffer lifetimes as approximate."
    )
    return ordered, "kernel_db_list_fallback", warn
