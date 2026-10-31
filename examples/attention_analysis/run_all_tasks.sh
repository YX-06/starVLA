#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source examples/attention_analysis/env.sh
run_label="${1:-alltasks_seed7}"
if (( $# )); then shift; fi
task_suites=("$@")
if (( ${#task_suites[@]} == 0 )); then
    task_suites=(libero_spatial libero_object libero_goal libero_10)
fi
if [[ ! "$run_label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Invalid run label: %s\n' "$run_label" >&2
    exit 2
fi
mkdir -p "$ATTENTION_ROOT/runs/tmux"
set -o noclobber
exec > "$ATTENTION_ROOT/runs/tmux/$run_label.log" 2>&1
"$ATTENTION_PYTHON" -m examples.attention_analysis.run_rollouts \
    --gpus 0 1 --task-suites "${task_suites[@]}" \
    --max-tasks -1 --episodes 1 --seed 7 --run-label "$run_label"
