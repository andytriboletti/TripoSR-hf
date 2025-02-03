#!/bin/bash

mkdir -p /logs

cd /workspace1 && python upper.py > /logs/log1.txt 2>&1 &
cd /workspace2 && python upper.py > /logs/log2.txt 2>&1 &
cd /workspace3 && python upper.py > /logs/log3.txt 2>&1 &
cd /workspace4 && python upper.py > /logs/log4.txt 2>&1 &

echo "All processes started in background."
