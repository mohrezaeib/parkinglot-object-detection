from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import math
from loguru import logger

from src.train.common import (
    split_source_from_yaml,
    labels_dir_for_split,
    list_image_stems,
    calc_gt_counts,
    count_mae_mse_rmse,
)

def yolov8_train_eval(
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
) -> Tuple[float, float]:
    from ultralytics import YOLO

    # 1) Train
    train_model = YOLO(base_weights or "yolov8n.pt")
    results = train_model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=project,
        name=fold_name,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        weight_decay=wd,
        pretrained=True,
        device=0,
        plots=False,
        verbose=False,
    )

    # 2) Resolve best.pt robustly
    save_dir = Path(getattr(results, "save_dir", Path(project) / fold_name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        alt = save_dir / "best.pt"
        if alt.exists():
            best = alt
        else:
            found = list(save_dir.rglob("best.pt"))
            if found:
                best = found[0]
            else:
                last = save_dir / "weights" / "last.pt"
                if last.exists():
                    best = last
                else:
                    raise FileNotFoundError(f"best.pt not found under {save_dir}")

    # 3) Reload best for evaluation
    eval_model = YOLO(str(best))

    # --- A) mAP with very low conf to avoid NaNs across versions ---
    conf_map = 0.001
    m = eval_model.val(data=data_yaml, imgsz=imgsz, conf=conf_map, iou=iou, split="val", save=False)
    map50 = getattr(getattr(m, "metrics", None), "map50", None)
    try:
        map50 = float(map50)
        if math.isnan(map50):
            map50 = 0.0
    except Exception:
        map50 = 0.0  # be safe

    # --- B) Counting with your chosen conf for NMS ---
    source, is_txt = split_source_from_yaml(data_yaml, split="val")
    preds = eval_model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou, stream=False, save=False, verbose=False)
    pred_counts = [len(r.boxes) for r in preds]

    labels_dir = labels_dir_for_split(data_yaml, split="val", is_kfold_val_txt=is_txt)
    stems = list_image_stems(source, is_txt)
    gt_counts = calc_gt_counts(labels_dir, stems)

    n = min(len(gt_counts), len(pred_counts))
    _, _, rmse = count_mae_mse_rmse(gt_counts[:n], pred_counts[:n])

    logger.info(f"\n\n\n[HPO/yv8] fold={fold_name} map50={map50 if map50==map50 else 'nan'} rmse={rmse:.3f}\n\n\n")
    return map50, rmse
