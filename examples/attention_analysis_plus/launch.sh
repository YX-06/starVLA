#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source examples/attention_analysis_plus/env.sh
label="${1:-plus_seed7_r1}"
shift || true
mkdir -p "$PLUS_ROOT/runs/tmux"
if [[ -e "$PLUS_ROOT/runs/$label" || -e "$PLUS_ROOT/runs/tmux/$label.log" ]]; then
    echo "Existing run or log for $label; choose a new label." >&2
    exit 1
fi
printf -v launch '%q ' "$PLUS_PYTHON" -u -m examples.attention_analysis_plus.run --label "$label" "$@"
printf -v log '%q' "$PLUS_ROOT/runs/tmux/$label.log"
printf -v repository '%q' "$PLUS_REPO"
tmux -S "/tmp/starvla-plus-$(id -u).sock" new-session -d -s "$label" \
  "cd $repository && source examples/attention_analysis_plus/env.sh && $launch > $log 2>&1"
echo "tmux: /tmp/starvla-plus-$(id -u).sock session $label"
echo "log: $PLUS_ROOT/runs/tmux/$label.log"
