import json
from pathlib import Path


def load_json(file_path):
    file = Path(file_path)
    return json.load(file)
