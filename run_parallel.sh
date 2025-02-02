#!/bin/bash
python /workspace1/upper.py & 
python /workspace2/upper.py & 
python /workspace3/upper.py & 
python /workspace4/upper.py &

# Store all background process IDs
pids=$!

# Wait for all processes
wait
#to stop all processes
#pkill -f upper.py
