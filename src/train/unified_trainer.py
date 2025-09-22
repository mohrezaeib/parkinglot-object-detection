# src/train/unified_trainer.py
from __future__ import annotations
from typing import Dict, Any
from .yolov8_trainer import run_fold_yolov8
from .yolo_legacy_trainer import run_fold_yolo_legacy

def run_one_fold(model_name: str, data_yaml: str, exp: Dict[str,Any],
                 model_cfg: Dict[str,Any], io_cfg: Dict[str,Any], fold_index: int) -> str:
    model = model_name.lower()
    if model == "yolov8":
        return run_fold_yolov8(data_yaml, exp, model_cfg, io_cfg, fold_index)   
    elif model == "yolov5":
        return run_fold_yolo_legacy("external/yolov5", data_yaml, exp, io_cfg, fold_index, model_name="yolov5")
    elif model == "yolov3":
        return run_fold_yolo_legacy("external/yolov3", data_yaml, exp, io_cfg, fold_index, model_name="yolov3")

    raise ValueError(f"Unsupported model: {model_name}")
