#!/bin/bash
# Attach a resumable 2k eval to an already-running training tmux session.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_TMUX_SOCKET="${TRAIN_TMUX_SOCKET:?Set TRAIN_TMUX_SOCKET}"
TRAIN_TMUX_SESSION="${TRAIN_TMUX_SESSION:?Set TRAIN_TMUX_SESSION}"
TRAIN_DCP="${TRAIN_DCP:?Set TRAIN_DCP}"
TRAIN_ITERS="${TRAIN_ITERS:?Set TRAIN_ITERS}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:?Set EVAL_OUTPUT_DIR}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
TRACKER="$TRAIN_DCP/latest_checkpointed_iteration.txt"

if ! [[ "$TRAIN_ITERS" =~ ^[0-9]+$ ]] || [ "$TRAIN_ITERS" -le 0 ]; then
    echo "TRAIN_ITERS must be a positive integer, got: $TRAIN_ITERS" >&2
    exit 1
fi
if ! [[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] || [ "$POLL_INTERVAL" -le 0 ]; then
    echo "POLL_INTERVAL must be a positive integer, got: $POLL_INTERVAL" >&2
    exit 1
fi

cd "$ROOT_DIR"
echo "Waiting for tmux $TRAIN_TMUX_SOCKET/$TRAIN_TMUX_SESSION at $(date --iso-8601=seconds)"

training_session_exists() {
    local session_name
    while IFS= read -r session_name; do
        if [ "$session_name" = "$TRAIN_TMUX_SESSION" ]; then
            return 0
        fi
    done < <(
        tmux -L "$TRAIN_TMUX_SOCKET" list-sessions -F '#{session_name}' 2>/dev/null || true
    )
    return 1
}

while training_session_exists; do
    sleep "$POLL_INTERVAL"
done

if [ ! -f "$TRACKER" ]; then
    echo "Training session ended without checkpoint tracker: $TRACKER" >&2
    exit 1
fi
LATEST_STEP="$(< "$TRACKER")"
if [ "$LATEST_STEP" != "$TRAIN_ITERS" ]; then
    echo "Training session ended at step $LATEST_STEP; expected $TRAIN_ITERS. Eval not started." >&2
    exit 1
fi

if [ -f "$EVAL_OUTPUT_DIR/summary.json" ]; then
    echo "Evaluation already complete: $EVAL_OUTPUT_DIR/summary.json"
    exit 0
fi

echo "Training reached step $TRAIN_ITERS; starting resumable 2k eval at $(date --iso-8601=seconds)"
TRAIN_DCP="$TRAIN_DCP" \
OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
bash fast_wam/scripts/eval_libero_trained_2k.sh
