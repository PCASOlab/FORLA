#!/bin/bash

# Federated Learning Tmux Setup Script

# Kill any existing tmux server to start fresh
# tmux kill-server  
 

# run client sessions for eval
tmux new-session -d -s client1 -n "FL Client1"
tmux send-keys -t client1 "python -m main.FL.FL_c1" C-m

tmux new-session -d -s client2 -n "FL Client2"
tmux send-keys -t client2 "python -m main.FL.FL_c2" C-m

tmux new-session -d -s client3 -n "FL Client3"
tmux send-keys -t client3 "python -m main.FL.FL_c3" C-m

tmux new-session -d -s client4 -n "FL Client4"
tmux send-keys -t client4 "python -m main.FL.FL_c4" C-m

# Optional: Attach to server session by default
tmux attach-session -t client1
