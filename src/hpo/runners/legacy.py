# src/hpo/runners/yolov3.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import yaml
from loguru import logger

from src.train.common import (
    split_source_from_yaml,     # (source, is_txt)
    labels_dir_for_split,       # GT labels dir (handles K-fold val TXT)
    list_image_stems,           # stems from TXT or folder
    calc_gt_counts,             # GT box counts per stem
    count_mae_mse_rmse,         # MAE/MSE/RMSE
)

# Adjust this if your yolov3 repo lives elsewhere
Y3_REPO = "/home/mohammad/object_detection/parkinglot-object-detection/external/yolov3"

# Reasonable default hyp file inside the repo
DEFAULT_HYP = "data/hyps/hyp.scratch-low.yaml"  # fallback if present


def _norm_optim_name(opt: Optional[str]) -> str:
    """Map flexible names to yolov3 CLI choices."""
    if not opt:
        return "SGD"
    o = opt.lower()
    if o == "adam":
        return "Adam"
    if o in ("adamw", "adam-w", "adam_w"):
        return "AdamW"
    return "SGD"


def _run(cmd: list[str], cwd: str, env: Optional[dict] = None) -> None:
    logger.info("Running: {} (cwd={})", " ".join(map(str, cmd)), cwd)
    subprocess.check_call(cmd, cwd=cwd, env=env)


def _latest_under(base: Path, pattern: str) -> Optional[Path]:
    candidates = list(base.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _latest_val_run_dir(repo: str) -> Optional[Path]:
    return _latest_under(Path(repo) / "runs" / "val", "exp*")


def _read_map50_from_val_csv(val_run_dir: Path) -> Optional[float]:
    csv = val_run_dir / "results.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        if len(df) == 0:
            return None
        if "metrics/mAP_0.5" in df.columns:
            return float(df["metrics/mAP_0.5"].iloc[-1])
        if "map50" in df.columns:
            return float(df["map50"].iloc[-1])
    except Exception:
        return None
    return None


def _count_predicted_boxes(val_run_dir: Path, stems: List[str]) -> List[int]:
    labels_dir = val_run_dir / "labels"
    counts: List[int] = []
    for s in stems:
        p = labels_dir / f"{s}.txt"
        n = 0
        if p.exists():
            with open(p, "r") as fh:
                n = sum(1 for _ in fh if _.strip())
        counts.append(n)
    return counts


def _write_hyp_file(repo: str, lr0: float, lrf: float, wd: float) -> str:
    """
    Create a temp hyp YAML by merging repo default with (lr0, lrf, weight_decay).
    Falls back to minimal dict if default hyp is missing.
    """
    base_hyp_path = Path(repo) / DEFAULT_HYP
    if base_hyp_path.exists():
        try:
            with open(base_hyp_path, "r") as f:
                hyp = yaml.safe_load(f) or {}
        except Exception:
            hyp = {}
    else:
        hyp = {}

    # Overwrite key params. YOLOv3/5 use these names in hyp files.
    hyp["lr0"] = float(lr0)
    hyp["lrf"] = float(lrf)
    hyp["weight_decay"] = float(wd)

    # Make sure some essentials exist (safe defaults if absent)
    hyp.setdefault("momentum", 0.937)
    hyp.setdefault("warmup_epochs", 3.0)
    hyp.setdefault("warmup_momentum", 0.8)
    hyp.setdefault("warmup_bias_lr", 0.1)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(hyp, tmp)
    tmp.close()
    return tmp.name


def yolov3_train_eval(
    base_weights: Optional[str],
    data_yaml: str,
    imgsz: int,
    epochs: int,
    batch: int,
    workers: int,
    optimizer: str,
    lr0: float,
    lrf: float,
    wd: float,
    conf: float,
    iou: float,
    project: str,
    fold_name: str,
    device: str = "0",
) -> Tuple[float, float]:
    """
    Train one fold in YOLOv3 repo and evaluate with val.py (no detect.py).
    Returns: (map50, rmse_count)
    """
    py = shutil.which("python") or "python"
    env = os.environ.copy()
    # Keep their own W&B silent; we log via our HPO process if needed
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"

    # 0) Create a custom hyp file with lr0/lrf/weight_decay for v3
    hyp_path = _write_hyp_file(Y3_REPO, lr0=lr0, lrf=lrf, wd=wd)

    # 1) Train (note: no --lr0/--lrf/--weight-decay on the CLI here)
    train_cmd = [
        py, "train.py",
        "--img", str(imgsz),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--data", str(Path(data_yaml).resolve()),
        "--project", str(project),
        "--name", str(fold_name),
        "--exist-ok",
        "--single-cls",
        "--save-period", "1",
        "--workers", str(workers),
        "--device", str(device),
        "--optimizer", _norm_optim_name(optimizer),
        "--hyp", str(hyp_path),
    ]
    if base_weights:
        train_cmd += ["--weights", str(base_weights)]

    _run(train_cmd, cwd=Y3_REPO, env=env)

    # 2) Find best.pt (prefer the intended folder, else latest anywhere under runs/)
    best = Path(Y3_REPO) / project / fold_name / "weights" / "best.pt"
    if not best.exists():
        alt = _latest_under(Path(Y3_REPO) / "runs", "**/weights/best.pt")
        if not alt or not alt.exists():
            raise FileNotFoundError(f"best.pt not found at {best}")
        best = alt

    # 3) Evaluate with val.py (saves labels in runs/val/exp*)
    val_cmd = [
        py, "val.py",
        "--data", str(Path(data_yaml).resolve()),
        "--weights", str(best),
        "--img-size", str(imgsz),                       # v3 uses --img-size
        "--conf-thres", str(max(conf, 0.001)),          # v3 warns if < 0.001
        "--iou-thres", str(iou),
        "--task", "val",
        "--single-cls",
        "--save-txt", "--save-conf",
        "--device", str(device),
        "--project", "runs/val",
        "--name", "exp",                                # repo will auto-increment
    ]
    _run(val_cmd, cwd=Y3_REPO, env=env)

    # 4) Parse results (mAP50) from latest val run
    val_run = _latest_val_run_dir(Y3_REPO)
    if val_run is None:
        raise FileNotFoundError(f"No val run directory found under {Y3_REPO}/runs/val")
    map50 = _read_map50_from_val_csv(val_run)
    if map50 is None:
        logger.warning("mAP50 not found in {}, defaulting to 0.0", val_run / "results.csv")
        map50 = 0.0

    # 5) Build stems in the correct order (matches our GT & val labels)
    source, is_txt = split_source_from_yaml(data_yaml, split="val")
    stems = list_image_stems(source, is_txt)

    # 6) Predicted counts from val labels
    pred_counts = _count_predicted_boxes(val_run, stems)

    # 7) GT counts and RMSE
    labels_dir = labels_dir_for_split(data_yaml, "val", is_kfold_val_txt=is_txt)
    gt_counts = calc_gt_counts(labels_dir, stems)
    _, _, rmse = count_mae_mse_rmse(gt_counts, pred_counts)

    logger.info("[HPO/yv3] fold={} map50={:.4f} rmse={:.3f}", fold_name, map50, rmse)

    # 8) Cleanup temp hyp
    try:
        os.remove(hyp_path)
    except Exception:
        pass

    return float(map50), float(rmse)
