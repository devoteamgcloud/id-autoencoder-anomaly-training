#!/bin/bash
set -e

# Generate args from config.json using inline Python
mapfile -t ARGS < <(
python3 - <<'PY'
import json

def kwargs_to_list(kwargs):
    args_list = []
    for key, value in kwargs.items():
        arg_key = f"--{key}"
        args_list.append(arg_key)
        if isinstance(value, list):
            args_list.extend(map(str, value))
        if isinstance(value, None):
            pass
        else:
            args_list.append(str(value))
    return args_list

with open('config.json', 'r') as f:
    kwargs = json.load(f)

for arg in kwargs_to_list(kwargs):
    print(arg)
PY
)

echo "Running task.py with args: ${ARGS[*]}"

python3 -m trainer.task "${ARGS[@]}"