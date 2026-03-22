"""Trace/kernel-database I/O and classification for the minimal MoE dataflow pass.

This is a minimal MoE-local reconstruction pass: it does not reconstruct the full
graph or infer exact tensor identity; it is intended only for architectural
intermediate-residency modeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.moe.detect import classify_kernel_entries


def load_kernel_database(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load ``kernel_database.json`` and return ``(root, kernels list)``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kernels = list(data.get("kernels") or [])
    return data, kernels


def classify_kernels_from_db(
    kernel_db: Dict[str, Any],
    kernels: List[Dict[str, Any]],
    *,
    shared_dim_override: Optional[int] = None,
    routed_dim_override: Optional[int] = None,
    moe_debug_log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run ``classify_kernel_entries`` using metadata from the DB root."""
    hf_config = (kernel_db.get("metadata") or {}).get("model_config")
    return classify_kernel_entries(
        kernels,
        model_config=hf_config,
        shared_dim_override=shared_dim_override,
        routed_dim_override=routed_dim_override,
        moe_debug_log_path=moe_debug_log_path,
    )
