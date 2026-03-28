"""MoE per-expert-type memory profiling pipeline.

Pass — NCU isolation (when `ncu` is available):
  ncu_profile_kernel() on sampled entries per expert type.
  Provides absolute HBM bytes (hardware counters, accurate regardless
  of cache context).  L1/L2 hit rates from NCU are isolation-only
  (self-reuse) and are labelled as such.

Usage::

    soda-cli --moe-profile --kernel-db-path <path>
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from soda.moe.detect import (
    classify_kernel_entries,
    moe_op_profile_debug_path,
)
from soda.moe.debug import debug_print
from soda.moe.op_profile import detect_num_layers_from_shared_patterns, generate_op_profile
from soda.moe.report import generate_moe_report

_DEFAULT_MAX_NCU_BUCKETS = 25
_DEFAULT_TARGET_HBM_COVERAGE = 0.85
_DEFAULT_MIN_BUCKETS_PER_ROLE = 1
_DEFAULT_LOW_VALUE_BUCKET_FRAC = 0.001
_DEFAULT_LOW_VALUE_BUCKET_SCALE = 0.01  # deprioritize low-value buckets


class MoEProfilePipeline:
    """MoE memory profiling: NCU isolation, trace-ordered execution_trace.json, aggregates in op_profile.json."""

    def __init__(self, kernel_db_path: Path, args) -> None:
        self.kernel_db_path = Path(kernel_db_path)
        self.args = args
        debug_print("MoEProfilePipeline.__init__", "kernel_db_path=", self.kernel_db_path)

        # Load kernel DB
        with open(self.kernel_db_path, "r", encoding="utf-8") as f:
            self.kernel_db = json.load(f)

        self.kernels: List[Dict] = self.kernel_db.get("kernels", [])

        self.trace_path = self.kernel_db_path.parent / "trace.json"
        debug_print("kernel_count=", len(self.kernels), "trace_path=", self.trace_path)

        # Derive output directory alongside the kernel DB
        self.output_dir = self.kernel_db_path.parent / "moe_profile"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        debug_print("output_dir=", self.output_dir)

        # Optional overrides
        self.shared_dim_override: Optional[int] = getattr(args, "moe_shared_dim", None)
        self.routed_dim_override: Optional[int] = getattr(args, "moe_routed_dim", None)
        debug_print(
            "overrides",
            "shared=", self.shared_dim_override,
            "routed=", self.routed_dim_override,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Path:
        """Run the full pipeline and return the path to moe_profile.json."""
        debug_print("run:start")
        print(f"\n[MoE Profile] Kernel DB: {self.kernel_db_path}")
        print(f"[MoE Profile] Trace:      {self.trace_path}")
        print(f"[MoE Profile] Output:     {self.output_dir}")

        if not self.trace_path.is_file():
            raise FileNotFoundError(
                f"MoE trace-centric profiling requires trace.json next to the kernel DB "
                f"(expected {self.trace_path})"
            )
        debug_print("trace_exists=", True)

        op_profile_path = self.output_dir / "op_profile.json"
        moe_debug_path = moe_op_profile_debug_path(op_profile_path)
        debug_print("op_profile_path=", op_profile_path, "moe_debug_path=", moe_debug_path)
        moe_debug_path.write_text(
            "# MoE debug log: kernel classification (detect) + op_profile reconstruction\n",
            encoding="utf-8",
        )

        # 1. Classify entries
        hf_config = self.kernel_db.get("metadata", {}).get("model_config")
        debug_print("classify:start")
        classified = classify_kernel_entries(
            self.kernels,
            model_config=hf_config,
            shared_dim_override=self.shared_dim_override,
            routed_dim_override=self.routed_dim_override,
            moe_debug_log_path=moe_debug_path,
        )
        debug_print("classify:done", "classified_count=", len(classified))
        self._print_classification_summary(classified)

        # 2. NCU isolation (HBM baseline)
        ncu_results: Dict[str, Dict] = {}
        debug_print("ncu_check:start")
        try:
            from soda.ncu import ncu_check_available
            if ncu_check_available():
                debug_print("ncu_check:available")
                ncu_results = self._run_ncu_pass(classified)
            else:
                print("[MoE Profile] ncu not available — skipping NCU pass")
                debug_print("ncu_check:unavailable")
        except ImportError:
            print("[MoE Profile] soda.ncu not importable — skipping NCU pass")
            debug_print("ncu_check:import_error")
        debug_print("ncu_pass:done", "ncu_result_count=", len(ncu_results))

        # 3. Aggregate report (NCU only)
        debug_print("report:start")
        report_path = generate_moe_report(
            classified_kernels=classified,
            ncu_results=ncu_results,
            output_dir=self.output_dir,
            args=self.args,
        )
        debug_print("report:done", "report_path=", report_path)
        print(f"\n[MoE Profile] Report: {report_path}")

        nsys_path = self.kernel_db_path.parent / "nsys_hbm" / "attribution_summary.json"
        if nsys_path.is_file():
            with open(report_path, "r", encoding="utf-8") as f:
                mrep = json.load(f)
            with open(nsys_path, "r", encoding="utf-8") as f:
                mrep["nsys_hbm_attribution_run"] = json.load(f)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(mrep, f, indent=2)

        # 4. execution_trace.json + aggregated op_profile.json (trace-ordered)
        debug_print("num_layers:detect:start")
        num_layers = self._get_num_layers(classified)
        debug_print("num_layers:detect:done", "num_layers=", num_layers)
        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        precision = cfg.get("precision", "bfloat16") or "bfloat16"
        debug_print("op_profile:start", "precision=", precision)
        profile = generate_op_profile(
            trace_path=self.trace_path,
            classified_kernels=classified,
            num_layers=num_layers,
            precision=precision,
            ncu_results=ncu_results,
            output_path=op_profile_path,
            execution_trace_path=self.output_dir / "execution_trace.json",
            moe_debug_log_path=moe_debug_path,
        )
        nrows = int(profile.get("row_count", 0))
        debug_print("op_profile:done", "row_count=", nrows)
        print(
            f"[MoE Profile] Execution trace ({nrows} GPU rows, {num_layers} layers): "
            f"{self.output_dir / 'execution_trace.json'}"
        )
        print(f"[MoE Profile] Op profile (aggregates): {op_profile_path}")
        print(f"[MoE Profile] MoE debug log: {moe_debug_path}")
        debug_print("run:done")

        return report_path

    # ------------------------------------------------------------------
    # NCU isolation
    # ------------------------------------------------------------------

    def _run_ncu_pass(self, classified: List[Dict]) -> Dict[str, Dict]:
        """Run an importance-ranked, bucketed NCU sampling pass.

        Instead of sampling entries uniformly, we:
          1) bucket classified entries into equivalence classes that preserve HBM behavior
          2) score buckets by estimated HBM bytes covered
          3) pick representative buckets under a budget/coverage policy
          4) profile one representative entry per selected bucket
          5) apply the measured bytes to all entries in that bucket

        Returns:
            Dict mapping kernel DB entry id -> NCU result dict (for op_profile overrides).
            Non-representative bucket entries are included only to propagate measured HBM.
        """
        from soda.common.data import clean_kernel_name
        from soda.moe.op_profile import _compute_hbm_fields, _dtype_bytes, _normalize_shape
        from soda.ncu import ncu_profile_kernel

        ncu_dir = self.output_dir / "ncu"
        ncu_dir.mkdir(parents=True, exist_ok=True)

        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        precision = cfg.get("precision", "bfloat16") or "bfloat16"
        dtype_bytes = _dtype_bytes(precision)

        max_ncu_buckets = int(getattr(self.args, "moe_ncu_max_buckets", _DEFAULT_MAX_NCU_BUCKETS) or _DEFAULT_MAX_NCU_BUCKETS)
        target_hbm_coverage = float(
            getattr(self.args, "moe_ncu_target_hbm_coverage", _DEFAULT_TARGET_HBM_COVERAGE)
            or _DEFAULT_TARGET_HBM_COVERAGE
        )
        min_buckets_per_role = int(
            getattr(self.args, "moe_ncu_min_buckets_per_role", _DEFAULT_MIN_BUCKETS_PER_ROLE)
            or _DEFAULT_MIN_BUCKETS_PER_ROLE
        )
        low_value_bucket_frac = float(
            getattr(self.args, "moe_ncu_low_value_bucket_frac", _DEFAULT_LOW_VALUE_BUCKET_FRAC)
            or _DEFAULT_LOW_VALUE_BUCKET_FRAC
        )
        low_value_bucket_scale = float(
            getattr(self.args, "moe_ncu_low_value_bucket_scale", _DEFAULT_LOW_VALUE_BUCKET_SCALE)
            or _DEFAULT_LOW_VALUE_BUCKET_SCALE
        )

        expert_types = ["shared_expert", "routed_expert", "gate"]
        required_structural_roles = [
            "shared_expert_expand",
            "shared_expert_down",
            "routed_expert_expand",
            "routed_expert_down",
            "moe_gate",
        ]

        def _shape_signature(input_dims: Any) -> str:
            """Stable normalized representation for bucket keys."""
            try:
                norm = _normalize_shape(input_dims)
                return ",".join(str(x) for x in norm)
            except Exception:
                return str(input_dims)

        def _estimate_entry_fields(entry: Dict) -> Tuple[float, float, int]:
            """Return (hbm_bytes_per_inv, avg_latency_us, frequency)."""
            stats = entry.get("statistics", {}) or {}
            freq = int(stats.get("frequency", 0) or 0)
            avg_dur_us = float(stats.get("avg_duration_us", 0.0) or 0.0)

            aten_name = entry.get("aten_op", {}).get("name", "") or ""
            input_dims = entry.get("aten_op", {}).get("input_dims", []) or []
            hbm = _compute_hbm_fields(aten_name, input_dims, dtype_bytes).get("hbm_bytes", 0.0)
            return float(hbm), float(avg_dur_us), freq

        # 1) Build NCU sampling buckets
        entry_buckets: Dict[
            Tuple[str, str, str, str, str],
            Dict[str, Any],
        ] = {}

        target_entries = [e for e in classified if e.get("expert_type") in expert_types]
        debug_print("ncu_pass:bucket:start", "target_entries=", len(target_entries), "precision=", precision)

        for entry in target_entries:
            entry_id = entry.get("id")
            if entry_id is None:
                continue

            et = entry.get("expert_type", "other")
            sr = entry.get("gemm_structural_role", "unknown_gemm")
            aten_name = entry.get("aten_op", {}).get("name", "") or ""
            cleaned_kernel = clean_kernel_name(entry.get("kernel", {}).get("name", "") or "")
            shape_sig = _shape_signature(entry.get("aten_op", {}).get("input_dims", []))

            key = (str(et), str(sr), str(aten_name), str(cleaned_kernel), str(shape_sig))

            hbm_per_inv, avg_latency_us, freq = _estimate_entry_fields(entry)

            b = entry_buckets.get(key)
            if b is None:
                b = {
                    "key": key,
                    "key_str": json.dumps(
                        {
                            "expert_type": et,
                            "structural_role": sr,
                            "aten_op": aten_name,
                            "cleaned_kernel": cleaned_kernel,
                            "shape_signature": shape_sig,
                        },
                        sort_keys=True,
                    ),
                    "expert_type": et,
                    "structural_role": sr,
                    "aten_op_name": aten_name,
                    "kernel_name_cleaned": cleaned_kernel,
                    "shape_signature": shape_sig,
                    "entries": [],  # list of entry dicts
                    "entry_data": [],  # list of lightweight per-entry info
                    "estimated_hbm_total": 0.0,
                    "estimated_latency_total_us": 0.0,
                    "estimated_frequency_total": 0,
                }
                entry_buckets[key] = b

            b["entries"].append(entry)
            b["entry_data"].append(
                {
                    "entry": entry,
                    "entry_id": str(entry_id),
                    "frequency": freq,
                    "avg_latency_us": avg_latency_us,
                    "hbm_per_inv": hbm_per_inv,
                }
            )
            b["estimated_hbm_total"] += hbm_per_inv * freq
            b["estimated_latency_total_us"] += avg_latency_us * freq
            b["estimated_frequency_total"] += freq

        buckets = list(entry_buckets.values())
        if not buckets:
            debug_print("ncu_pass:bucket:empty")
            return {}

        total_relevant_hbm = sum(b.get("estimated_hbm_total", 0.0) for b in buckets) or 0.0
        max_bucket_hbm = max((b.get("estimated_hbm_total", 0.0) for b in buckets), default=0.0) or 0.0

        debug_print(
            "ncu_pass:bucket:built",
            "num_buckets=", len(buckets),
            "total_relevant_hbm=", total_relevant_hbm,
            "max_bucket_hbm=", max_bucket_hbm,
        )

        # 2) Representative selection within each bucket
        def _choose_representative(bucket: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            entries_data = list(bucket.get("entry_data", []))
            if not entries_data:
                return None

            # Prefer non-degenerate entries (non-zero estimated HBM).
            nonzero = [d for d in entries_data if (d.get("hbm_per_inv") or 0.0) > 0.0]
            pool = nonzero if nonzero else entries_data

            max_freq = max(d.get("frequency", 0) for d in pool)
            top = [d for d in pool if int(d.get("frequency", 0) or 0) == int(max_freq)]
            if not top:
                top = pool

            # If frequencies tie, prefer median-latency; if latency ties too, pick median-shape.
            if len(top) == 1:
                return top[0]

            # Drop tiny/degen within the tie set (avoid selecting an outlier with hbm_per_inv ~ 0).
            top_max_hbm = max((d.get("hbm_per_inv") or 0.0) for d in top)
            top_filtered = [d for d in top if (d.get("hbm_per_inv") or 0.0) >= top_max_hbm * 0.05]
            if top_filtered:
                top = top_filtered

            lat_sorted = sorted(top, key=lambda d: float(d.get("avg_latency_us", 0.0) or 0.0))
            lat_values = [float(d.get("avg_latency_us", 0.0) or 0.0) for d in lat_sorted]
            # If all latencies are effectively identical, fall back to median shape ordering.
            if lat_values and max(lat_values) - min(lat_values) < 1e-9:
                # Shape signature is fixed for a bucket key, but keep a deterministic fallback.
                lat_sorted = sorted(top, key=lambda d: str(bucket.get("shape_signature", "")))

            return lat_sorted[len(lat_sorted) // 2]

        for b in buckets:
            rep = _choose_representative(b)
            if rep is not None:
                b["representative_entry"] = rep["entry"]
                b["representative_entry_id"] = rep["entry_id"]
            else:
                b["representative_entry"] = None
                b["representative_entry_id"] = None

        # 3) Bucket scoring + budget/coverage selection
        buckets_by_role: Dict[str, List[Dict[str, Any]]] = {}
        for b in buckets:
            buckets_by_role.setdefault(b.get("structural_role", "unknown_gemm"), []).append(b)
        role_priority = {"routed_expert": 3, "shared_expert": 2, "gate": 1}
        for role, blist in buckets_by_role.items():
            blist.sort(
                key=lambda x: (
                    float(x.get("estimated_hbm_total", 0.0) or 0.0),
                    float(x.get("estimated_latency_total_us", 0.0) or 0.0),
                    float(x.get("estimated_frequency_total", 0) or 0),
                    int(role_priority.get(x.get("expert_type"), 0) or 0),
                ),
                reverse=True,
            )

        # Ensure max_ncu_buckets supports the mandatory set (otherwise take as many as possible).
        mandatory_target = len(required_structural_roles) * max(1, min_buckets_per_role)
        if max_ncu_buckets < mandatory_target:
            print(
                "[MoE Profile] Warning: moe_ncu_max_buckets is smaller than the mandatory "
                f"role coverage ({max_ncu_buckets} < {mandatory_target}). "
                "Only a subset of mandatory roles will be included."
            )

        selected: List[Dict[str, Any]] = []
        selected_keys: set = set()
        selected_hbm_total = 0.0

        # Always include at least one representative bucket per important role.
        for role in required_structural_roles:
            role_buckets = buckets_by_role.get(role, [])
            if not role_buckets:
                debug_print("ncu_pass:mandatory_missing", "role=", role)
                continue
            for b in role_buckets[: max(1, min_buckets_per_role)]:
                if len(selected) >= max_ncu_buckets:
                    break
                k = b["key"]
                if k in selected_keys:
                    continue
                selected.append(b)
                selected_keys.add(k)
                selected_hbm_total += float(b.get("estimated_hbm_total", 0.0) or 0.0)

        if not total_relevant_hbm:
            debug_print("ncu_pass:bucket:total_hbm_zero")
            return {}

        def _coverage(hbm_total: float) -> float:
            return max(0.0, min(1.0, hbm_total / total_relevant_hbm))

        current_cov = _coverage(selected_hbm_total)

        remaining = [b for b in buckets if b["key"] not in selected_keys]
        def _effective_importance(b: Dict[str, Any]) -> float:
            est = float(b.get("estimated_hbm_total", 0.0) or 0.0)
            deprioritize = (
                max_bucket_hbm > 0.0
                and est < max_bucket_hbm * low_value_bucket_frac
                and b.get("structural_role") not in required_structural_roles
            )
            return est * (low_value_bucket_scale if deprioritize else 1.0)

        remaining.sort(
            key=lambda x: (
                _effective_importance(x),
                float(x.get("estimated_hbm_total", 0.0) or 0.0),
                float(x.get("estimated_latency_total_us", 0.0) or 0.0),
                float(x.get("estimated_frequency_total", 0) or 0),
                int(role_priority.get(x.get("expert_type"), 0) or 0),
            ),
            reverse=True,
        )

        # Greedily add buckets by importance until reaching NCU budget or coverage target.
        for b in remaining:
            if len(selected) >= max_ncu_buckets:
                break
            if current_cov >= target_hbm_coverage:
                break
            selected.append(b)
            selected_keys.add(b["key"])
            selected_hbm_total += float(b.get("estimated_hbm_total", 0.0) or 0.0)
            current_cov = _coverage(selected_hbm_total)

        print(
            "[MoE Profile] NCU bucket sampling: "
            f"roles={required_structural_roles}, max_buckets={max_ncu_buckets}, "
            f"target_coverage={target_hbm_coverage:.2f}, selected={len(selected)}, "
            f"estimated_coverage={current_cov:.2%} (HBM bytes)"
        )
        debug_print(
            "ncu_pass:selection",
            "selected_buckets=", len(selected),
            "selected_cov=", current_cov,
        )

        # 7) Save bucket-to-representative mapping
        sampling_payload: List[Dict[str, Any]] = []
        for b in selected:
            sampling_payload.append(
                {
                    "bucket_key": b.get("key_str"),
                    "expert_type": b.get("expert_type"),
                    "structural_role": b.get("structural_role"),
                    "aten_op_name": b.get("aten_op_name"),
                    "kernel_name_cleaned": b.get("kernel_name_cleaned"),
                    "shape_signature": b.get("shape_signature"),
                    "representative_entry_id": b.get("representative_entry_id"),
                    "bucket_entry_count": len(b.get("entries", [])),
                    "estimated_hbm_bytes_total": round(float(b.get("estimated_hbm_total", 0.0) or 0.0), 2),
                    "estimated_latency_total_us": round(float(b.get("estimated_latency_total_us", 0.0) or 0.0), 2),
                    "estimated_frequency_total": int(b.get("estimated_frequency_total", 0) or 0),
                    "ncu_profile_success": False,
                }
            )

        bucket_map_path = self.output_dir / "ncu_bucket_sampling.json"
        bucket_map_path.write_text(json.dumps(sampling_payload, indent=2), encoding="utf-8")

        # 5) Profile NCU only representatives, apply measured bytes to whole bucket.
        results: Dict[str, Dict] = {}
        rep_count = 0
        for i, b in enumerate(selected, 1):
            rep_entry = b.get("representative_entry")
            rep_entry_id = b.get("representative_entry_id")
            if rep_entry is None or rep_entry_id is None:
                debug_print("ncu_pass:skip_no_rep", "bucket_i=", i, "bucket=", b.get("key_str"))
                continue

            op = b.get("aten_op_name", "?")
            kname = b.get("kernel_name_cleaned", "?")
            print(
                f"[MoE Profile] NCU bucket [{i}/{len(selected)}]: "
                f"role={b.get('structural_role')} expert={b.get('expert_type')} rep={rep_entry_id} "
                f"{op} -> {kname} (entries={len(b.get('entries', []))})"
            )
            debug_print("ncu_pass:bucket_rep", "bucket_key=", b.get("key_str"), "rep=", rep_entry_id)

            rep_count += 1
            result = ncu_profile_kernel(rep_entry, output_dir=ncu_dir)
            if result is None:
                debug_print("ncu_pass:bucket_rep_failed", "rep_entry_id=", rep_entry_id)
                continue

            metrics = result.get("metrics", {}) or {}
            # Remap raw dram__* metrics to the friendly names op_profile/report expect.
            result_mapped: Dict[str, Any] = dict(result)
            result_mapped["hbm_read_bytes"] = float(metrics.get("dram__bytes_read.sum", 0.0) or 0.0)
            result_mapped["hbm_write_bytes"] = float(metrics.get("dram__bytes_write.sum", 0.0) or 0.0)
            result_mapped["l1_hit_rate_pct"] = metrics.get("l1tex__t_sector_hit_rate.pct")
            result_mapped["l2_hit_rate_pct"] = metrics.get("lts__t_sector_hit_rate.pct")
            result_mapped["compute_util_pct"] = metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            )

            # Apply to every entry in the bucket so op_profile can override estimated bytes.
            for ed in b.get("entry_data", []):
                eid = ed.get("entry_id")
                if eid is None:
                    continue
                mapped = dict(result_mapped)
                mapped["kernel_id"] = str(eid)
                mapped["expert_type"] = b.get("expert_type")
                mapped["is_representative"] = str(eid) == str(rep_entry_id)
                mapped["representative_entry_id"] = str(rep_entry_id)
                mapped["bucket_key"] = b.get("key_str")
                mapped["bucket_entry_count"] = int(len(b.get("entries", [])) or 0)
                mapped["bucket_estimated_hbm_bytes_total"] = float(b.get("estimated_hbm_total", 0.0) or 0.0)
                results[str(eid)] = mapped

            # Update the saved sampling_payload success flag for this bucket.
            # (We don't re-parse the JSON file; just update local structure then rewrite at end.)
            for sp in sampling_payload:
                if sp.get("bucket_key") == b.get("key_str"):
                    sp["ncu_profile_success"] = True
                    break

        bucket_map_path.write_text(json.dumps(sampling_payload, indent=2), encoding="utf-8")

        debug_print("ncu_pass:done", "result_count=", len(results), "rep_count=", rep_count)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_num_layers(self, classified: List[Dict]) -> int:
        """Determine number of transformer layers for op_profile layer expansion.

        Priority:
          1. CLI --moe-num-layers override.
          2. num_hidden_layers from HuggingFace AutoConfig (if model accessible).
          3. GCD-based detection from shared expert entry frequencies.
        """
        # CLI override takes highest priority.
        num_layers_override = getattr(self.args, "moe_num_layers", None)
        debug_print("num_layers:override=", num_layers_override)
        if num_layers_override is not None:
            return int(num_layers_override)

        # Try HuggingFace AutoConfig.
        meta = self.kernel_db.get("metadata", {})
        cfg = meta.get("config", meta)
        model_name = cfg.get("model_name") or cfg.get("model")
        if model_name:
            try:
                from transformers import AutoConfig
                debug_print("num_layers:hf_try", "model_name=", model_name)
                hf_cfg = AutoConfig.from_pretrained(model_name)
                num_layers = getattr(hf_cfg, "num_hidden_layers", None)
                if num_layers:
                    debug_print("num_layers:hf_success", "num_layers=", num_layers)
                    return int(num_layers)
            except Exception:
                debug_print("num_layers:hf_failed_fallback")
                pass

        # GCD-based fallback on kernel DB shared-expert frequencies.
        debug_print("num_layers:gcd_fallback")
        return detect_num_layers_from_shared_patterns(classified)

    @staticmethod
    def _print_classification_summary(classified: List[Dict]) -> None:
        from collections import Counter
        counts = Counter(e.get("expert_type", "other") for e in classified)
        print("\n[MoE Profile] Classification summary:")
        for et in ["shared_expert", "routed_expert", "gate", "attention", "other"]:
            n = counts.get(et, 0)
            if n > 0:
                print(f"  {et:<18} {n:>5} entries")
        print()
