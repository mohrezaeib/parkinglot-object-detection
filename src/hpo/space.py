# src/hpo/space.py
from __future__ import annotations
import optuna

def suggest_core(trial: optuna.Trial, model: str):
    imgsz = trial.suggest_categorical("imgsz", [512, 640, 768])
    batch = trial.suggest_categorical("batch", [8, 16, 32])

    if model == "yolov8":
        optimizer = trial.suggest_categorical("optimizer", ["auto", "sgd", "adamw"])
    else:
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])

    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 5e-4, 5e-2, log=True)
    wd  = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

    return dict(imgsz=imgsz, batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf, weight_decay=wd)

def suggest_aug(trial: optuna.Trial):
    return dict(
        hsv_h=trial.suggest_float("hsv_h", 0.0, 0.015),
        hsv_s=trial.suggest_float("hsv_s", 0.4, 0.9),
        hsv_v=trial.suggest_float("hsv_v", 0.2, 0.7),
        degrees=trial.suggest_float("degrees", 0.0, 5.0),
        scale=trial.suggest_float("scale", 0.5, 1.5),
        shear=trial.suggest_float("shear", 0.0, 1.0),
        mixup=trial.suggest_float("mixup", 0.0, 0.2),
        mosaic=trial.suggest_float("mosaic", 0.5, 1.0),
    )
