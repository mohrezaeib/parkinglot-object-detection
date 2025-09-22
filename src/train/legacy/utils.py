# src/train/legacy/utils.py
from __future__ import annotations
import time
from pathlib import Path

def file_is_stable(p: Path, wait_ms: int = 800) -> bool:
    """True if file exists and size stops changing across a short window."""
    if not p.exists():
        return False
    s1 = p.stat().st_size
    time.sleep(wait_ms / 1000.0)
    if not p.exists():
        return False
    s2 = p.stat().st_size
    return s1 > 0 and s1 == s2
