# src/train/legacy/spawn.py
from __future__ import annotations
import os, shutil, subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

def spawn_train_external(
    repo: str,
    data_yaml: str,
    exp: Dict[str, Any],
    io_cfg: Dict[str, Any],
    run_name: str,
    extra: Optional[list[str]] = None,
) -> subprocess.Popen:
    py = shutil.which("python") or "python"
    cmd = [
        py, "train.py",
        "--img", str(exp["imgsz"]),
        "--epochs", str(exp["epochs"]),
        "--batch", str(exp["batch"]),
        "--data", str(Path(data_yaml).resolve()),
        "--project", str(io_cfg["project"]),
        "--name", run_name,
        "--exist-ok",
        "--single-cls",
        "--save-period", "1",
        "--workers", str(io_cfg.get("workers", 4)),
    ]
    if io_cfg.get("weights"):
        cmd += ["--weights", io_cfg["weights"]]
    if extra:
        cmd += extra

    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    logger.info(f"Launching {' '.join(cmd)} in {repo}")
    # STREAM stdout so we can detect end-of-epoch
    return subprocess.Popen(
        cmd, cwd=repo, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True
    )
