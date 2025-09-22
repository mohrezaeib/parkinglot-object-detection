# src/train/common.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import yaml
import numpy as np

# ---------- YAML helpers ----------
def load_yaml(p: str):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def split_source_from_yaml(data_yaml: str, split: str) -> Tuple[str, bool]:
    """
    Returns (source, is_txt). If YAML's split is a TXT file, return that path and True.
    Otherwise return images/<split> under dataset root and False.
    """
    d = load_yaml(data_yaml)
    if "path" not in d:
        raise ValueError(f"'path' missing in dataset yaml: {data_yaml}")
    if split == "val" and isinstance(d.get("val"), str) and d["val"].endswith(".txt"):
        return d["val"], True
    root = Path(d["path"])
    return str(root / f"images/{split}"), False

def labels_dir_for_split(data_yaml: str, split: str, is_kfold_val_txt: bool) -> Path:
    """
    GT labels:
      - Normal: labels/<split>
      - K-fold val via TXT: labels/train (since val is drawn from train pool)
    """
    d = load_yaml(data_yaml)
    if "path" not in d:
        raise ValueError(f"'path' missing in dataset yaml: {data_yaml}")
    base = Path(d["path"])
    if split == "val" and is_kfold_val_txt:
        return base / "labels/train"
    return base / f"labels/{split}"

# ---------- IO helpers ----------
def list_image_stems(source: str, is_txt: bool) -> List[str]:
    """
    If is_txt=True, read absolute image paths from TXT (one per line) and return stems.
    Else, list images under 'source' directory and return stems.
    """
    stems: List[str] = []
    if is_txt:
        with open(source, "r") as fh:
            for line in fh:
                p = line.strip()
                if p:
                    stems.append(Path(p).stem)
    else:
        exts = {".jpg", ".jpeg", ".png"}
        for p in sorted(Path(source).glob("*")):
            if p.suffix.lower() in exts:
                stems.append(p.stem)
    return stems

def calc_gt_counts(labels_dir: Path, stems: List[str]) -> List[int]:
    counts: List[int] = []
    for s in stems:
        lp = labels_dir / f"{s}.txt"
        n = 0
        if lp.exists():
            with open(lp, "r") as fh:
                n = sum(1 for _ in fh if _.strip())
        counts.append(n)
    return counts

# ---------- metrics ----------
def count_mae_mse_rmse(gt: List[int], pd: List[int]) -> Tuple[float, float, float]:
    gt_arr = np.array(gt, dtype=float); pd_arr = np.array(pd, dtype=float)
    if gt_arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    diff = pd_arr - gt_arr
    mae  = float(np.mean(np.abs(diff)))
    mse  = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse
