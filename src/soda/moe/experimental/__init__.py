"""Experimental MoE dataflow reconstruction (simulator-facing).

This package intentionally bypasses the flat ``op_profile`` row model.  Callers
should use :func:`experimental_pipeline.run_experimental_moe_dataflow` after
kernel-database classification, or rely on ``MoEProfilePipeline`` which invokes
it when experimental export is enabled.
"""

from soda.moe.experimental.experimental_pipeline import run_experimental_moe_dataflow

__all__ = ["run_experimental_moe_dataflow"]
