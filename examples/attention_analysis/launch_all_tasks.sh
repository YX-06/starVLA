#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
run_label="${1:-alltasks_seed7}"
if (( $# )); then shift; fi
if [[ ! "$run_label" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    printf 'Invalid run label: %s\n' "$run_label" >&2
    exit 2
fi
repo=$(pwd)
socket="/tmp/starvla-attn-$(id -u).sock"
printf -v command 'bash %q %q' "$repo/examples/attention_analysis/run_all_tasks.sh" "$run_label"
for suite in "$@"; do
    printf -v argument ' %q' "$suite"
    command+="$argument"
done
if tmux -S "$socket" has-session -t starvla-attn 2>/dev/null; then
    tmux -S "$socket" new-window -d -t starvla-attn -n "$run_label" "$command"
else
    tmux -S "$socket" new-session -d -s starvla-attn -n "$run_label" "$command"
fi
printf 'Run launched: %s\nAttach: tmux -S %s attach -t starvla-attn\n' "$run_label" "$socket"
