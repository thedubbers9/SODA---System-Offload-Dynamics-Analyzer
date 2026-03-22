"""Minimal MoE routed-expert chain reconstruction for architectural buffer modeling.

This package does **not** reconstruct the full computation graph, infer exact tensor
identity, or model shared experts. It only extracts short gate→select/metadata→
paired ``_grouped_mm`` chains and emits logical R/M/P/E/D buffers with approximate
sizes for intermediate-residency estimation.
"""

from soda.moe.moe_dataflow.main import run_minimal_moe_dataflow

__all__ = ["run_minimal_moe_dataflow"]
