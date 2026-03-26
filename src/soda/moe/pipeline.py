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
from typing import Dict, List, Optional

from soda.moe.detect import (
    classify_kernel_entries,
    get_entries_by_type,
    moe_op_profile_debug_path,
    sample_routed_entries,
)
from soda.moe.debug import debug_print
from soda.moe.op_profile import detect_num_layers_from_shared_patterns, generate_op_profile
from soda.moe.report import generate_moe_report

_NCU_SAMPLE_SIZE = 10  # Max entries to NCU-profile per expert type


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
        """Run NCU on sampled entries per expert type.

        Returns dict mapping kernel_id -> NCU result dict.
        """
        from soda.ncu import ncu_profile_kernel

        ncu_dir = self.output_dir / "ncu"
        ncu_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Dict] = {}
        expert_types = ["shared_expert", "routed_expert", "gate"]
        debug_print("ncu_pass:start", "expert_types=", expert_types)

        for et in expert_types:
            entries = get_entries_by_type(classified, et)
            if not entries:
                debug_print("ncu_pass:type", et, "entries=0")
                continue

            if et == "routed_expert":
                to_profile = sample_routed_entries(entries, n_samples=_NCU_SAMPLE_SIZE)
            else:
                to_profile = entries[:_NCU_SAMPLE_SIZE]

            debug_print(
                "ncu_pass:type",
                et,
                "entries=", len(entries),
                "to_profile=", len(to_profile),
            )

            print(
                f"[MoE Profile] NCU: {et} — profiling {len(to_profile)} "
                f"of {len(entries)} entries"
            )

            for i, entry in enumerate(to_profile, 1):
                kid = entry.get("id", f"?{i}")
                op = entry.get("aten_op", {}).get("name", "?")
                kname = entry.get("kernel", {}).get("name", "?")
                print(f"  [{i}/{len(to_profile)}] {kid}: {op} -> {kname}")
                debug_print("ncu_pass:kernel", "id=", kid, "op=", op, "kernel=", kname)

                result = ncu_profile_kernel(entry, output_dir=ncu_dir)
                if result is not None:
                    result["expert_type"] = et
                    results[kid] = result
                debug_print("ncu_pass:kernel_done", "id=", kid, "has_result=", result is not None)

        debug_print("ncu_pass:done", "result_count=", len(results))
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
