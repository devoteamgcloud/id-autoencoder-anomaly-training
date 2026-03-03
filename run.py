import json
import subprocess

def kwargs_to_list(kwargs):
    """Converts dict to flat list format: ['--arg', 'val', '--list', 'item1', 'item2']"""
    args_list = []
    for key, value in kwargs.items():
        arg_key = f"--{key}"
        args_list.append(arg_key)
        if isinstance(value, list):
            args_list.extend(map(str, value))
        else:
            args_list.append(str(value))
    return args_list

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        kwargs = json.load(f)

    args = kwargs_to_list(kwargs)

    # Run: python3 task.py ...args
    subprocess.run(["python3", "task.py", *args], check=True)