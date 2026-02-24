from datetime import datetime
import re
import argparse

def int_or_float(value):
    try:
        return int(value)
    except ValueError:
        try:
            val = float(value)
            if 0.0 < val <= 1.0:
                return float(value)
            elif val > 1.0:
                return int(round(value))
            raise ValueError('Can\'t be negative')
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")

def valid_bq_path(value):
    try:
        project, dataset, table = re.findall(r'^(.*)\.(.*)\.(.*)$', value)[0]
        return value
    except ValueError:
        raise argparse.ArgumentTypeError(f"bq path should be in `project.dataset.table` format. Found `{value}`.")


def valid_datetime(value):
    if value == '':
        return datetime.now().date()
    try:
        date = datetime.strptime(value, '%Y-%m-%d').date()
        return date
    except ValueError:
        raise argparse.ArgumentTypeError(f"date should be in YYYY-MM-DD format. found {value}.")


def valid_postfix(value):
    if value == '':
        return value
    if not value.startswith('_'):
        value  = '_' + value
        
    # Only allow alphanumeric characters without underscore (except at the beginning)
    if not re.match(r'^_[a-zA-Z0-9]+$', value):
        raise argparse.ArgumentTypeError(f"postfix should only contain alphanumeric characters and start with an underscore. found {value}.")
    return value