"""Interval overlap helpers for trace / profiler attribution."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def compute_overlap_ns(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Length of overlap of [a_start, a_end) and [b_start, b_end) in nanoseconds."""
    if a_end <= a_start or b_end <= b_start:
        return 0
    lo = max(a_start, b_start)
    hi = min(a_end, b_end)
    return max(0, hi - lo)


def attribute_window_to_intervals(
    w_start_ns: int,
    w_end_ns: int,
    intervals: Sequence[Dict],
    start_key: str = "start_ns",
    end_key: str = "end_ns",
) -> Tuple[List[Tuple[int, float]], float]:
    """
    Split one sample window across overlapping intervals proportional to overlap duration.

    Args:
        w_start_ns, w_end_ns: window [start, end) in ns.
        intervals: list of dicts with start_key, end_key (ns).
    Returns:
        (list of (interval_index, share), unassigned_fraction)
        share sums to 1.0 over overlapping intervals; unassigned_fraction is 1 - sum(shares)
        when no overlap or partial coverage.
    """
    overlaps: List[Tuple[int, int]] = []
    for idx, it in enumerate(intervals):
        ist = int(it.get(start_key, 0) or 0)
        ien = int(it.get(end_key, 0) or 0)
        ov = compute_overlap_ns(w_start_ns, w_end_ns, ist, ien)
        if ov > 0:
            overlaps.append((idx, ov))
    denom = sum(o for _, o in overlaps)
    if denom <= 0:
        return [], 1.0
    shares = [(idx, ov / denom) for idx, ov in overlaps]
    return shares, 0.0
