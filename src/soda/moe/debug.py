"""Minimal debug logger for MoE profiling code paths.

"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any


def _debug_enabled() -> bool:
    return True


def debug_print(*parts: Any) -> None:
    """Print debug messages when SODA_DEBUG is enabled."""
    if not _debug_enabled():
        return
    ts = datetime.now().isoformat(timespec="seconds")
    msg = " ".join(str(p) for p in parts)
    print(f"[SODA DEBUG {ts}] {msg}")
