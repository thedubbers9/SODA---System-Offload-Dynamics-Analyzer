#!/bin/bash
# shellcheck shell=bash
# Path helpers shared by SLURM templates and local sweep scripts.
#
# Provides:
#   soda_make_output_root <output_tag>
#
# Behavior:
# - Uses SODA_OUTPUT as base if set, otherwise defaults to "$SODA_ROOT/output".
# - Returns a unique directory name: "<tag>_job<id>_<YYYYmmdd_HHMMSS>".
# - Prints the path to stdout (no extra text), suitable for command substitution.

_soda_abs_path() {
    local p="$1"
    if [[ -z "$p" ]]; then
        return 1
    fi
    if [[ "$p" = /* ]]; then
        printf "%s\n" "$p"
        return 0
    fi
    printf "%s/%s\n" "$PWD" "$p"
}

soda_output_base_dir() {
    local base="${SODA_OUTPUT:-}"
    if [[ -z "$base" ]]; then
        if [[ -z "${SODA_ROOT:-}" ]]; then
            echo "error: neither SODA_OUTPUT nor SODA_ROOT is set" >&2
            return 1
        fi
        base="$SODA_ROOT/output"
    fi
    _soda_abs_path "$base"
}

soda_make_output_root() {
    local output_tag="${1:-run}"
    local base
    base="$(soda_output_base_dir)" || return 1

    # Keep tag filename-safe and deterministic.
    output_tag="${output_tag//[^a-zA-Z0-9._-]/_}"
    if [[ -z "$output_tag" ]]; then
        output_tag="run"
    fi

    local job_id="${SLURM_JOB_ID:-local}"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"

    printf "%s/%s_job%s_%s\n" "$base" "$output_tag" "$job_id" "$ts"
}
