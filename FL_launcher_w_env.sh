#!/bin/bash

# Federated Learning Tmux Setup Script

# Ensure conda is initialized for tmux sessions
# (adjust path to your conda.sh if necessary)
source ~/anaconda3/etc/profile.d/conda.sh

# Create server session
tmux new-session -d -s server -n "FL Server"
tmux send-keys -t server "conda activate forla && python -m main.FL.FL_server" C-m

# Create client sessions
tmux new-session -d -s client1 -n "FL Client1"
tmux send-keys -t client1 "conda activate forla && python -m main.FL.FL_c1" C-m

tmux new-session -d -s client2 -n "FL Client2"
tmux send-keys -t client2 "conda activate forla && python -m main.FL.FL_c2" C-m

tmux new-session -d -s client3 -n "FL Client3"
tmux send-keys -t client3 "conda activate forla && python -m main.FL.FL_c3" C-m

tmux new-session -d -s client4 -n "FL Client4"
tmux send-keys -t client4 "conda activate forla && python -m main.FL.FL_c4" C-m

# Optional: Attach to server session by default
tmux attach-session -t server