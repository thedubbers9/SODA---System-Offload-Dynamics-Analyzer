#!/bin/bash
# =============================================================================
# SODA output root helper
# =============================================================================
# Some SLURM templates expect `soda_make_output_root` to exist after sourcing
# this file. It returns (prints) a unique output directory path; the caller
# is responsible for creating it (`mkdir -p`).
#
# Example:
#   source "$SODA_ROOT/slurm/output_paths.sh"
#   OUTPUT_ROOT="$(soda_make_output_root "my-tag")"
#   mkdir -p "$OUTPUT_ROOT"

set -euo pipefail

soda_make_output_root() {
  local tag="${1:-soda}"
  local job_id="${SLURM_JOB_ID:-local}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"

  local base="${SODA_OUTPUT:-"$PWD/output"}"
  # Keep it simple + stable across shells: one line of output to stdout.
  echo "${base%/}/${tag}_job${job_id}_${ts}"
}

