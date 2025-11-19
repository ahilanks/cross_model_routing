#!/bin/bash

# config
BASE_SESSION_NAME="router_training"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SESSION_NAME="${BASE_SESSION_NAME}"
PYTHON_SCRIPT="router_acc_only.py" #change here if diff script

LOG_FILE="logs/${SESSION_NAME}_${TIMESTAMP}.log"

echo "Starting tmux session '$SESSION_NAME'..."
echo "Logs will be saved to: $LOG_FILE"

TRAIN_CMD="python -u $PYTHON_SCRIPT 2>&1 | tee $LOG_FILE"

# Start tmux detached, running your training
tmux new-session -d -s $SESSION_NAME "$TRAIN_CMD"

# Attach to it
tmux attach-session -t $SESSION_NAME

echo "Session '$SESSION_NAME' is running."
