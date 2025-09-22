# src/eval/yolov8_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import wandb
from ultralytics import YOLO
from src.train.common import (
    split_source_from_yaml,    
    labels_dir_for_split,      
    list_image_stems,          
    calc_gt_counts,                 
    count_mae_mse_rmse,        
)

def eval_yolov8(weights: str, data: str, imgsz: int, split: str, conf: float, iou: float,
                save_vis: bool, wb_project: str, wb_name: Optional[str]) -> dict:
    wandb.init(project=wb_project, name=wb_name or f"eval-{Path(weights).stem}", reinit=True)
    model = YOLO(weights)

    # 1) mAP via Ultralytics val()
    m = model.val(data=data, imgsz=imgsz, conf=conf, iou=iou, split=split, save=save_vis)
    map50 = float(getattr(m, "metrics", {}).get("map50", float("nan")))

    # 2) Count metrics on split (TXT or folder)
    source, is_txt = split_source_from_yaml(data, split)
    preds = model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou, stream=False, save=save_vis)
    pred_counts = [len(r.boxes) for r in preds]

    labels_dir = labels_dir_for_split(data, split, is_kfold_val_txt=is_txt)
    stems = list_image_stems(source, is_txt)
    gt = calc_gt_counts(labels_dir, stems)

    n = min(len(gt), len(pred_counts))
    mae, mse, rmse = count_mae_mse_rmse(gt[:n], pred_counts[:n])

    wandb.log({
        "eval/map50": map50,
        "eval/count_mae": mae,
        "eval/count_mse": mse,
        "eval/count_rmse": rmse,
        "eval/num_images": n,
        "eval/conf": conf,
        "eval/iou": iou,
        "eval/split": split,
    })
    out = {"map50": map50, "MAE": mae, "MSE": mse, "RMSE": rmse, "N": n}
    print(json.dumps(out, indent=2))
    wandb.finish()
    return out
