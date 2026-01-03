from typing import List, Dict
from pathlib import Path
import hashlib
import json

def load_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_text(path: str) -> str:
    """Reads a file from the configured input directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding='utf-8')


def compute_hash(content: str) -> str:
    """Returns SHA256 hash of the content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
