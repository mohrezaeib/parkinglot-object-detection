# src/hpo/runners.py
from __future__ import annotations
import os, shutil, subprocess, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from loguru import logger

from src.train.kfold import ensure_kfold_splits
from src.train.fold_yaml import create_fold_yaml
from src.utils.datasets import load_yaml
from src.train.common import (
    split_source_from_yaml, labels_dir_for_split, list_image_stems,
    calc_gt_counts, count_mae_mse_rmse
)

# --- legacy helpers (YOLOv5/3) ---
def legacy_train_one_fold(model: str, repo: str, data_yaml: str, imgsz: int, epochs: int, batch: int,
                          lr0: float, lrf: float, wd: float, optimizer: str, project: str,
                          fold_name: str, workers: int, base_weights: Optional[str]) -> str:
    py = shutil.which("python") or "python"
    cmd = [
        py, "train.py",
        "--img", str(imgsz),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--data", str(Path(data_yaml).resolve()),
        "--project", project,
        "--name", fold_name,
        "--exist-ok", "--single-cls",
        "--save-period", "0",
        "--workers", str(workers),
    ]
    if base_weights:
        cmd += ["--weights", base_weights]
    if model == "yolov5":
        cmd += ["--optimizer", optimizer, "--lr0", str(lr0), "--lrf", str(lrf), "--weight-decay", str(wd)]
    env = os.environ.copy(); env["WANDB_DISABLED"] = "true"
    logger.info(f"[HPO] Launching: {' '.join(cmd)} (cwd={repo})")
    subprocess.check_call(cmd, cwd=repo, env=env)
    best = Path(project) / fold_name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found in {best.parent}")
    return str(best)

def legacy_val_map(repo: str, data_yaml: str, weights: str, imgsz: int, conf: float, iou: float) -> Optional[float]:
    py = shutil.which("python") or "python"
    env = os.environ.copy(); env["WANDB_DISABLED"] = "true"
    for s in ["val.py", "test.py"]:
        if (Path(repo) / s).exists():
            try:
                subprocess.check_call([
                    py, s, "--data", str(Path(data_yaml).resolve()),
                    "--weights", str(weights), "--img", str(imgsz),
                    "--conf", str(conf), "--iou", str(iou),
                    "--task", "val", "--single-cls",
                ], cwd=repo, env=env)
            except subprocess.CalledProcessError:
                pass
            break
    try:
        import pandas as pd
        csvs = sorted(Path(repo).glob("runs/**/results.csv"))
        if not csvs: return None
        df = pd.read_csv(csvs[-1])
        if "metrics/mAP_0.5" in df.columns: return float(df.iloc[-1]["metrics/mAP_0.5"])
        if "map50" in df.columns: return float(df.iloc[-1]["map50"])
    except Exception:
        return None
    return None

def legacy_detect_counts(repo: str, weights: str, source: str, imgsz: int, conf: float, iou: float) -> List[int]:
    py = shutil.which("python") or "python"
    out_name = f"hpo_eval_{int(time.time()*1000)}"
    cmd = [
        py, "detect.py", "--weights", str(weights), "--img", str(imgsz),
        "--conf", str(conf), "--iou", str(iou), "--source", str(source),
        "--save-txt", "--save-conf", "--project", "runs/hpo_eval", "--name", out_name, "--exist-ok",
    ]
    env = os.environ.copy(); env["WANDB_DISABLED"] = "true"
    subprocess.check_call(cmd, cwd=repo, env=env)
    labels_dir = Path(repo) / "runs" / "hpo_eval" / out_name / "labels"
    stems = list_image_stems(source, is_txt=source.endswith(".txt"))
    out = []
    for s in stems:
        p = labels_dir / f"{s}.txt"; n = 0
        if p.exists():
            with open(p, "r") as fh: n = sum(1 for _ in fh if _.strip())
        out.append(n)
    return out

# --- yolov8 one-fold runner ---
def yolov8_train_eval(base_weights: Optional[str], data_yaml: str, imgsz: int, epochs: int, batch: int,
                      workers: int, optimizer: str, lr0: float, lrf: float, wd: float,
                      conf: float, iou: float, project: str, fold_name: str) -> Tuple[float, float]:
    from ultralytics import YOLO
    model = YOLO(base_weights or "yolov8n.pt")
    results = model.train(
        data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch, project=project, name=fold_name,
        workers=workers, optimizer=optimizer, lr0=lr0, lrf=lrf, weight_decay=wd,
        pretrained=True, device=0, plots=False
    )
    best = Path(results.save_dir) / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found at {best}")
    m = model.val(weights=str(best), data=data_yaml, imgsz=imgsz, conf=conf, iou=iou, split="val", save=False)
    map50 = float(getattr(m, "metrics", {}).get("map50", float("nan")))
    source, is_txt = split_source_from_yaml(data_yaml, "val")
    preds = model.predict(weights=str(best), source=source, imgsz=imgsz, conf=conf, iou=iou, stream=False, save=False)
    pred_counts = [len(r.boxes) for r in preds]
    labels_dir = labels_dir_for_split(data_yaml, "val", is_kfold_val_txt=is_txt)
    stems = list_image_stems(source, is_txt)
    gt = calc_gt_counts(labels_dir, stems)
    _, _, rmse = count_mae_mse_rmse(gt, pred_counts)
    return map50, rmse

# --- fold orchestration ---
def kfold_iter(data_yaml: str, folds_to_eval: int, seed: int):
    dataset_root = Path(load_yaml(data_yaml)["path"])
    splits_dir = ensure_kfold_splits(dataset_root, folds_to_eval, seed)
    for k in range(folds_to_eval):
        yield k, create_fold_yaml(data_yaml, splits_dir / f"fold{k}_train.txt", splits_dir / f"fold{k}_val.txt")
