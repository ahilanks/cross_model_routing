#!/bin/bash

# config
BASE_SESSION_NAME="router_training"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
SESSION_NAME="${BASE_SESSION_NAME}"
PYTHON_SCRIPT="router_final.py"
TENSORBOARD_LOGDIR="runs"
TENSORBOARD_PORT="6006"

LOG_FILE="logs/${SESSION_NAME}_${TIMESTAMP}.log"

echo "Starting tmux session '$SESSION_NAME'..."
echo "Logs will be saved to: $LOG_FILE"

TRAIN_CMD="python -u $PYTHON_SCRIPT 2>&1 | tee $LOG_FILE"
tmux new-session -d -s $SESSION_NAME "$TRAIN_CMD"

tmux split-window -h -t $SESSION_NAME

tmux send-keys -t $SESSION_NAME.1 "tensorboard --logdir=$TENSORBOARD_LOGDIR --port=$TENSORBOARD_PORT" C-m

tmux select-pane -t $SESSION_NAME.0

tmux attach-session -t $SESSION_NAME

echo "Session '$SESSION_NAME' is running."