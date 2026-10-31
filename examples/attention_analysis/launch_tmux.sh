#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo=$(pwd)
socket="/tmp/starvla-attn-$(id -u).sock"
command="bash '$repo/examples/attention_analysis/run_background.sh'"
if tmux -S "$socket" has-session -t starvla-attn 2>/dev/null; then
    tmux -S "$socket" new-window -d -t starvla-attn -n libero "$command"
else
    tmux -S "$socket" new-session -d -s starvla-attn -n libero "$command"
fi
printf 'LIBERO pipeline launched. Attach with:\ntmux -S %s attach -t starvla-attn\n' "$socket"
