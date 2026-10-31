#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source examples/attention_analysis/env.sh
mkdir -p "$ATTENTION_ROOT/runs/tmux"
trap '"$ATTENTION_PYTHON" -m examples.attention_analysis.report' EXIT
"$ATTENTION_PYTHON" -u -m examples.attention_analysis.background 2>&1 | tee -a "$ATTENTION_ROOT/runs/tmux/pipeline.log"
