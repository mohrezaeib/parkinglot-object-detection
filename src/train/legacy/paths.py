# src/train/legacy/paths.py
from __future__ import annotations
import time
from pathlib import Path

def unique_suffix() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def resolve_external_run_dir(repo: str, project: str, run_name: str) -> Path:
    proj_path = Path(project)
    base = proj_path if proj_path.is_absolute() else (Path(repo) / proj_path)
    return base / run_name

def detect_output_labels_dir(repo: str, project: str, name: str) -> Path:
    """
    Convenience helper in case you standardize detect outputs.
    Currently unused because detect module computes based on flags.
    """
    return Path(repo) / project / name / "labels"
