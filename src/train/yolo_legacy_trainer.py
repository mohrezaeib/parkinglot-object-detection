# src/train/yolo_legacy_trainer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from src.train.legacy.paths import unique_suffix, resolve_external_run_dir
from src.train.legacy.spawn import spawn_train_external
from src.train.legacy.poller import stream_and_log_counts

def run_fold_yolo_legacy(
    repo: str,
    data_yaml: str,
    exp: Dict[str, Any],
    io_cfg: Dict[str, Any],
    fold_index: int,
    model_name: str = "yolov5",  # or "yolov3"
) -> str:
    suffix = unique_suffix()
    run_name = f"fold{fold_index}_{suffix}"

    # spawn training
    proc = spawn_train_external(repo, data_yaml, exp, io_cfg, run_name)
    run_dir = resolve_external_run_dir(repo, io_cfg["project"], run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[legacy] Run dir resolved to: {run_dir}")

    try:
        stream_and_log_counts(
            proc=proc,
            repo=repo,
            run_dir=run_dir,
            data_yaml=data_yaml,
            exp=exp,
            unique_suffix=suffix,
            model_name=model_name,   # "yolov5" | "yolov3"
            step_start=0,
        )
    finally:
        proc.wait()

    return str(run_dir / "weights" / "best.pt")
