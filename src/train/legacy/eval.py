# src/train/legacy/eval.py
from __future__ import annotations
import os, shutil, subprocess, time, tempfile
from pathlib import Path
from typing import List, Tuple
from loguru import logger

def _cmd_val_v5(py: str, weights: str, data_yaml: str, imgsz: int, conf: float, iou: float, device:str="0") -> list[str]:
    return [
        py, "val.py",
        "--data", str(Path(data_yaml).resolve()),
        "--weights", weights,
        "--img", str(imgsz),
        "--conf", str(conf),
        "--iou", str(iou),
        "--task", "val",
        "--single-cls",
        "--save-txt",
        "--save-conf",
        "--device", device,          # <— warning  GPU contention
    ]

def _cmd_val_v3(py: str, weights: str, data_yaml: str, imgsz: int, conf: float, iou: float,  device:str="0") -> list[str]:
    return [
        py, "val.py",
        "--data", str(Path(data_yaml).resolve()),
        "--weights", weights,
        "--img-size", str(imgsz),
        "--conf-thres", str(conf),
        "--iou-thres", str(iou),
        "--task", "val",
        "--single-cls",
        "--save-txt",
        "--save-conf",
        "--device", device,          # <— warning GPU contention
    ]

def _run(cmd: list[str], cwd: str) -> Tuple[bool, str]:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    return (r.returncode == 0), out

def _latest_val_labels_dir(repo: str) -> Path:
    base = Path(repo) / "runs" / "val"
    if not base.exists():
        return Path()
    candidates = sorted([p for p in base.glob("**/labels") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else Path()

def _copy_weights_to_tmp(weights: Path) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_legacy_weights"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_w = tmp_dir / f"{weights.stem}_{int(time.time()*1000)}.pt"
    shutil.copy2(weights, tmp_w)   # <— atomic read source for val.py
    return tmp_w

def run_repo_val_and_count(
    repo: str,
    model_name: str,          # "yolov5" | "yolov3"
    weights: Path,            # checkpoint to evaluate
    data_yaml: str,           # fold yaml
    stems: List[str],         # stems in eval order (from your TXT or folder)
    imgsz: int,
    conf: float,
    iou: float,
    device:str
) -> List[int]:
    """
    Runs the external repo's val.py to generate predictions on the val split defined
    in `data_yaml`, then returns per-image counts aligned to `stems`.
    """

    py = shutil.which("python") or "python"

    # Copy the checkpoint to a temp file to avoid concurrent read while training writes
    tmp_weights = _copy_weights_to_tmp(weights)

    tried_cmds = []
    if model_name.lower() == "yolov5":
        tried_cmds = [_cmd_val_v5(py, str(tmp_weights), data_yaml, imgsz, conf, iou,device),
                      _cmd_val_v3(py, str(tmp_weights), data_yaml, imgsz, conf, iou, device)]
    else:
        tried_cmds = [_cmd_val_v3(py, str(tmp_weights), data_yaml, imgsz, conf, iou, device),
                      _cmd_val_v5(py, str(tmp_weights), data_yaml, imgsz, conf, iou, device)]

    # Up to 3 retries with exponential backoff in case the file just got flushed
    last_out = ""
    ok = False
    for attempt in range(3):
        for cmd in tried_cmds:
            ok, out = _run(cmd, cwd=repo)
            if ok:
                break
            logger.warning(f"[legacy] val.py failed (attempt {attempt+1}/3) with cmd: {' '.join(cmd)}")
            logger.warning(out.strip()[:2000])
            last_out = out
        if ok:
            break
        time.sleep(1.0 * (attempt + 1))  # backoff

    if not ok:
        raise subprocess.CalledProcessError(returncode=1, cmd=tried_cmds[-1], output=last_out)

    labels_out = _latest_val_labels_dir(repo)
    if not labels_out.exists():
        raise FileNotFoundError("val labels directory not found under runs/val")

    pred_counts: List[int] = []
    for s in stems:
        p = labels_out / f"{s}.txt"
        n = 0
        if p.exists():
            with open(p, "r") as fh:
                n = sum(1 for _ in fh if _.strip())
        pred_counts.append(n)
    return pred_counts
