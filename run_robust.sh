#!/bin/bash
set -euo pipefail

# --- Configuration ---
MODEL=$1
FEATURES=$2
PIPELINE=${3:-individual}     # pipeline type (individual|global)
OVERWRITE_FLAG=""             # optional --overwrite

# Backward compatibility for old 3-arg usage: ./run_robust.sh MODEL FEATURES --overwrite
if [ "$PIPELINE" = "--overwrite" ]; then
  OVERWRITE_FLAG="--overwrite"
  PIPELINE="individual"
else
  # If a 4th arg exists and is --overwrite, set it
  if [ "${4:-}" = "--overwrite" ]; then
    OVERWRITE_FLAG="--overwrite"
  fi
fi

# Validate args
if [ -z "$MODEL" ] || [ -z "$FEATURES" ]; then
    echo "Usage: ./run_robust.sh <model> <features> [pipeline: individual|global] [--overwrite]"
    echo "Examples:"
    echo "  ./run_robust.sh LSTM Speed"
    echo "  ./run_robust.sh LSTM Speed global --overwrite"
    exit 1
fi

case "$PIPELINE" in
  individual|global) ;;
  *)
    echo "Invalid pipeline '$PIPELINE'. Expected 'individual' or 'global'."
    exit 1
    ;;
esac

# Memory-based restart configuration
MEM_THRESHOLD_GIB=14       # restart when used RAM >= this (GiB)
CHECK_INTERVAL_SEC=5       # how often to poll memory
GRACE_SHUTDOWN_SEC=20      # wait this long after SIGINT before SIGKILL

# Optional safety fallback (0 disables walltime cap)
MAX_WALLTIME_SEC=0         # e.g. 1800 for 30 minutes, or 0 to disable

# --- Argument Validation ---
if [ -z "$MODEL" ] || [ -z "$FEATURES" ]; then
    echo "Usage: ./run_robust.sh <model> <features> [--overwrite]"
    echo "Example: ./run_robust.sh LSTM Speed"
    exit 1
fi

# --- Helpers ---
used_mem_kib() {
  # Returns total used memory in KiB (MemTotal - MemAvailable)
  awk '
    /MemTotal:/ {t=$2}
    /MemAvailable:/ {a=$2}
    END {print (t - a)}
  ' /proc/meminfo
}

format_gib() {
  # KiB -> GiB (string with 2 decimals)
  awk -v kib="$1" 'BEGIN {printf "%.2f", kib/1024/1024}'
}

is_running() {
  kill -0 "$1" 2>/dev/null
}

group_alive() { pgrep -g "$1" >/dev/null 2>&1; }  # any proc in PGID?

kill_group_gracefully() {
  local PGID="$1"
  local GRACE="$2"
  echo "Stopping process group PGID=$PGID (grace ${GRACE}s)..."
  # SIGINT first
  kill -SIGINT -"$PGID" 2>/dev/null || true
  for _ in $(seq 1 "$GRACE"); do
    group_alive "$PGID" || return 0
    sleep 1
  done
  # SIGTERM
  kill -TERM -"$PGID" 2>/dev/null || true
  for _ in $(seq 1 "$GRACE"); do
    group_alive "$PGID" || return 0
    sleep 1
  done
  # SIGKILL
  echo "Forcing termination of PGID=$PGID..."
  kill -KILL -"$PGID" 2>/dev/null || true
  # Busy-wait briefly until group disappears
  for _ in $(seq 1 5); do
    group_alive "$PGID" || return 0
    sleep 1
  done
  return 0
}

# Convert GiB threshold to KiB for integer comparisons
THRESHOLD_KIB=$(( MEM_THRESHOLD_GIB * 1024 * 1024 ))

# --- Outer loop to execute all iterations ---
for ITERATION in 1 2 3 4 5; do
  echo "===================================================="
  echo "Starting Iteration $ITERATION | Model: $MODEL | Features: $FEATURES | Pipeline: $PIPELINE"
  echo "===================================================="

  FINAL_METRICS_FILE="results/${PIPELINE}/iteration_${ITERATION}/${MODEL}/${FEATURES}/final_metrics.json"

  if [ -f "$FINAL_METRICS_FILE" ]; then
    echo "SUCCESS: Results for Iteration $ITERATION already exist. Skipping."
    continue
  fi

  # Robust inner loop for the current iteration
  while [ ! -f "$FINAL_METRICS_FILE" ]; do
    echo "----------------------------------------------------"
    echo "STARTING RUN at $(date)"
    echo "Iteration: $ITERATION, Model: $MODEL, Features: $FEATURES, Pipeline: $PIPELINE"
    echo "Memory threshold: ${MEM_THRESHOLD_GIB} GiB | Check every ${CHECK_INTERVAL_SEC}s"
    [ "$MAX_WALLTIME_SEC" -gt 0 ] && echo "Walltime safety cap: ${MAX_WALLTIME_SEC}s"
    echo "----------------------------------------------------"

    # Start Python in a NEW SESSION/GROUP so we can kill its entire tree
    # -u for unbuffered output to keep logs timely
    PY_MODULE="src.pipeline_${PIPELINE}"
    setsid python3 -u -m "$PY_MODULE" --iteration "$ITERATION" --model "$MODEL" --features "$FEATURES" $OVERWRITE_FLAG &
    PY_PID=$!
    PGID=$PY_PID
    echo "Started Python PID=$PY_PID PGID=$PGID"

    START_TS=$(date +%s)
    EXIT_CODE=0
    RESTART_REASON="completed"

    # Monitor loop
    while kill -0 "$PY_PID" 2>/dev/null; do
      # Memory check
      USED_KIB=$(used_mem_kib)
      if [ "$USED_KIB" -ge "$THRESHOLD_KIB" ]; then
        USED_GIB=$(format_gib "$USED_KIB")
        echo "MEMORY THRESHOLD EXCEEDED: ${USED_GIB} GiB >= ${MEM_THRESHOLD_GIB} GiB. Restarting..."
        RESTART_REASON="memory"
        kill_group_gracefully "$PGID" "$GRACE_SHUTDOWN_SEC"
        EXIT_CODE=124
        break
      fi

      # Optional walltime safety cap
      if [ "$MAX_WALLTIME_SEC" -gt 0 ]; then
        NOW=$(date +%s)
        ELAPSED=$(( NOW - START_TS ))
        if [ "$ELAPSED" -ge "$MAX_WALLTIME_SEC" ]; then
          echo "Walltime cap reached (${ELAPSED}s). Restarting..."
          RESTART_REASON="walltime"
          kill -SIGINT "$PY_PID" 2>/dev/null || true
          for _ in $(seq 1 "$GRACE_SHUTDOWN_SEC"); do
            is_running "$PY_PID" || break
            sleep 1
          done
          if is_running "$PY_PID"; then
            echo "Forcing termination..."
            kill -9 "$PY_PID" 2>/dev/null || true
          fi
          EXIT_CODE=124
          break
        fi
      fi

      sleep "$CHECK_INTERVAL_SEC"
    done

    # If parent exited by itself, get exit code
    if ! kill -0 "$PY_PID" 2>/dev/null && [ "$EXIT_CODE" -eq 0 ]; then
      wait "$PY_PID" || EXIT_CODE=$?
      RESTART_REASON="completed"
    fi

    # Handle exit codes
    if [ $EXIT_CODE -eq 0 ]; then
      echo "Python script completed successfully."
      break
    elif [ $EXIT_CODE -eq 124 ]; then
      echo "Normal restart due to: $RESTART_REASON"
      # Ensure no straggler workers from the killed group
      if group_alive "$PGID"; then
        echo "Warning: residual processes in PGID=$PGID; forcing kill."
        kill -KILL -"$PGID" 2>/dev/null || true
      fi
    elif [ $EXIT_CODE -eq 137 ]; then
      echo "Process killed by OOM (137). Treating as memory restart."
    else
      echo "ERROR: Python failed with exit $EXIT_CODE. Aborting."
      exit 1
    fi

    echo "----------------------------------------------------"
    echo "RUN CYCLE COMPLETE at $(date). Preparing to restart..."
    echo "----------------------------------------------------"
    sleep 5
  done

  # Check final metrics
  if [ -f "$FINAL_METRICS_FILE" ]; then
    echo "----------------------------------------------------"
    echo "SUCCESS: Final metrics file found for Iteration $ITERATION."
    echo "----------------------------------------------------"
  fi
done

echo "===================================================="
echo "ALL ITERATIONS COMPLETE for the $PIPELINE Pipeline, Model $MODEL, Features $FEATURES."
echo "===================================================="