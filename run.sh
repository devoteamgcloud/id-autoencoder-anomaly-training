#!/bin/bash
set -e

ARGS=$(python3 run.py)

echo "Running task.py with args: $ARGS"

python3 trainer/task.py $ARGS