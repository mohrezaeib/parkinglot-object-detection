# src/train/legacy/poller.py
from __future__ import annotations
import time, glob
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import wandb

from src.train.common import (
    split_source_from_yaml,
    labels_dir_for_split,
    list_image_stems,
    calc_gt_counts,
    count_mae_mse_rmse,
)
from .eval import run_repo_val_and_count
from .epoch_stream import iter_epoch_end_lines

def _select_ckpt_for_epoch(weights_dir: Path, epoch_idx: int) -> Path | None:
    """Prefer epoch{n}.pt if present; fallback to last.pt; else best.pt."""
    ep = weights_dir / f"epoch{epoch_idx}.pt"
    if ep.exists():
        return ep
    last = weights_dir / "last.pt"
    if last.exists():
        return last
    best = weights_dir / "best.pt"
    if best.exists():
        return best
    return None

def stream_and_log_counts(
    proc,                   # Popen with stdout=PIPE
    repo: str,
    run_dir: Path,
    data_yaml: str,
    exp: Dict[str, Any],
    unique_suffix: str,
    model_name: str,
    step_start: int = 0,
):
    if wandb.run is None:
        wandb.init(project="ParkingLotOD", name=f"legacy-eval-{unique_suffix}", reinit=True)

    # Prepare GT once
    source, is_txt = split_source_from_yaml(data_yaml, split="val")
    stems = list_image_stems(source, is_txt)
    labels_dir = labels_dir_for_split(data_yaml, split="val", is_kfold_val_txt=is_txt)
    gt_count = calc_gt_counts(labels_dir, stems)

    weights_dir = run_dir / "weights"
    logger.info(f"[legacy] Streaming epochs; weights at: {weights_dir}")

    eval_device = exp.get("eval_device", "cpu")
    eval_batch  = int(exp.get("eval_batch", 8))
    eval_half   = bool(exp.get("eval_half", False))

    # Read the training output and react at epoch end
    for epoch_idx in iter_epoch_end_lines(proc.stdout):
        ckpt = _select_ckpt_for_epoch(weights_dir, epoch_idx)
        if not ckpt:
            logger.warning(f"[legacy] No checkpoint found for epoch {epoch_idx}; skipping eval for this epoch.")
            continue

        try:
              pred_counts = run_repo_val_and_count(
                repo=repo,
                model_name=model_name,
                weights=ckpt,
                data_yaml=data_yaml,
                stems=stems,
                imgsz=exp["imgsz"],
                conf=exp["conf_thres"],
                iou=exp["iou_thres"],
                device="0",
            )
        except Exception as e:
            logger.warning(f"[legacy] Epoch {epoch_idx}: eval failed ({e}); skipping.")
            continue

        mae, mse, rmse = count_mae_mse_rmse(gt_count, pred_counts)
        wandb.log({"val/count_mae": mae, "val/count_mse": mse, "val/count_rmse": rmse},
                  step=epoch_idx + step_start)
        logger.info(f"[legacy] Epoch {epoch_idx}: MAE={mae:.3f} MSE={mse:.3f} RMSE={rmse:.3f}")

    # After training stops, you can add an optional final best.pt eval:
    best = weights_dir / "best.pt"
    if best.exists():
        try:
            
            pred_counts = run_repo_val_and_count(
                repo=repo,
                model_name=model_name,
                weights=ckpt,
                data_yaml=data_yaml,
                stems=stems,
                imgsz=exp["imgsz"],
                conf=exp["conf_thres"],
                iou=exp["iou_thres"],
                device="0",
            )
            mae, mse, rmse = count_mae_mse_rmse(gt_count, pred_counts)
            wandb.log({"val/best_count_mae": mae, "val/best_count_mse": mse, "val/best_count_rmse": rmse})
            logger.info(f"[legacy] Final(best): MAE={mae:.3f} MSE={mse:.3f} RMSE={rmse:.3f}")
        except Exception as e:
            logger.warning(f"[legacy] Final(best) eval failed: {e}")
