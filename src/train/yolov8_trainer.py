# src/train/yolov8_trainer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import wandb
from ultralytics import YOLO
from src.train.common import (
    count_mae_mse_rmse,
    calc_gt_counts,                # <- corrected helper name
    labels_dir_for_split,
    list_image_stems,
    split_source_from_yaml,
)
from loguru import logger

def run_fold_yolov8(
    data_yaml: str,
    exp: Dict[str, Any],
    model_cfg: Dict[str, Any],
    io_cfg: Dict[str, Any],
    fold_index: int,
) -> str:
    # Build training model from weights or default
    base_weights = model_cfg.get("weights") or "yolov8n.pt"
    yolo_train = YOLO(base_weights)

    # Precompute GT counts for this fold’s val (works for TXT or folder)
    source, is_txt = split_source_from_yaml(data_yaml, split="val")
    stems = list_image_stems(source, is_txt)
    labels_dir = labels_dir_for_split(data_yaml, split="val", is_kfold_val_txt=is_txt)
    gt_count = calc_gt_counts(labels_dir, stems)

    # Callback: after each fit epoch, Ultralytics has saved trainer.last (last.pt)
    # Use a fresh model for prediction to avoid interfering with the training graph.
    def on_fit_epoch_end(trainer):
        ckpt_path = getattr(trainer, "last", None)  # path to the latest 'last.pt'
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.warning(
                f"[YOLOv8] Skipping per-epoch count eval at epoch {epoch}: "
                f"checkpoint not found (trainer.last={ckpt_path!r})."
            )
            # optional: mark it in W&B so you can see skipped epochs
            wandb.log({"val/eval_skipped": 1}, step=epoch)
            return
        yolo_eval = YOLO(str(ckpt_path))
        preds = yolo_eval.predict(
            source=source,
            imgsz=exp["imgsz"],
            conf=exp["conf_thres"],
            iou=exp["iou_thres"],
            stream=False,
            save=False,
            verbose=False,
        )
        pred_counts = [len(r.boxes) for r in preds]
        mae, mse, rmse = count_mae_mse_rmse(gt_count, pred_counts)
        epoch = int(getattr(trainer, "epoch", 0))
        wandb.log(
            {
                "val/count_mae": mae,
                "val/count_mse": mse,
                "val/count_rmse": rmse,
            },
            step=epoch,
        )

    # Register callback (avoid using on_val_end with predict() on the live model)
    yolo_train.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Train
    results = yolo_train.train(
        data=data_yaml,
        imgsz=exp["imgsz"],
        epochs=exp["epochs"],
        batch=exp["batch"],
        project=io_cfg["project"],
        name=f"fold{fold_index}",
        workers=io_cfg.get("workers", 4),
        optimizer=model_cfg.get("optimizer", "auto"),
        lr0=model_cfg["lr0"],
        lrf=model_cfg["lrf"],
        weight_decay=model_cfg["weight_decay"],
        pretrained=True,
        seed=exp.get("seed", 42),
        device=0,
        plots=True,
        amp=True,  # explicit (defaults to True on GPU)
    )

    best = Path(results.save_dir) / "best.pt"
    return str(best)
