# src/hpo/objective.py
from __future__ import annotations
from typing import Dict, Any, Callable
import numpy as np
import optuna
import wandb
from loguru import logger

from src.hpo.space import suggest_core, suggest_aug
from src.hpo.score import composite_score
from src.hpo.runners import yolov8_train_eval, yolov3_train_eval

from src.hpo.runners.utils import kfold_iter
from src.train.common import (
    split_source_from_yaml,
    labels_dir_for_split,
    list_image_stems,
    calc_gt_counts,
)

def make_objective(base_cfg: Dict[str, Any], args) -> Callable[[optuna.Trial], float]:
    """
    Builds an Optuna objective that:
      - samples core + augmentation params,
      - runs k-fold training/eval,
      - returns a single scalar via composite_score(map50, rmse, gt_counts).
    """
    def objective(trial: optuna.Trial) -> float:
        model = args.model.lower()
        data_yaml = base_cfg["dataset"]["yaml"]
        conf = float(base_cfg["experiment"]["conf_thres"])
        iou = float(base_cfg["experiment"]["iou_thres"])
        seed = int(base_cfg["experiment"].get("seed", 42))
        k_all = int(base_cfg["dataset"].get("folds", 5))
        folds_to_eval = min(getattr(args, "folds", k_all), k_all)

        # --- search space ---
        core = suggest_core(trial, model)   # imgsz, batch, optimizer (for v5), lr0, lrf, weight_decay
        _aug = suggest_aug(trial)           # sampled but not directly consumed here (your training code may read from YAML)
        epochs = int(args.epochs)

        # --- W&B per trial ---
        wb = wandb.init(
            project=base_cfg["wandb"]["project"],
            entity=base_cfg["wandb"]["entity"],
            name=f"HPO-{model}-t{trial.number}",
            reinit=True,
            config={**core, **_aug, "epochs": epochs, "conf": conf, "iou": iou, "model": model},
        )

        per_fold_scores, per_fold_map, per_fold_rmse = [], [], []

        for k, fold_yaml in kfold_iter(data_yaml, folds_to_eval, seed):
            fold_name = f"trial{trial.number}_fold{k}"

            # For normalization inside composite_score we need GT counts once per fold
            src, is_txt = split_source_from_yaml(fold_yaml, "val")
            stems = list_image_stems(src, is_txt)
            labels_dir = labels_dir_for_split(fold_yaml, "val", is_kfold_val_txt=is_txt)
            gtc = calc_gt_counts(labels_dir, stems)

            if model == "yolov8":
                # Train + eval yolov8
                map50, rmse = yolov8_train_eval(
                    base_weights=base_cfg["model"].get("weights"),
                    data_yaml=fold_yaml,
                    imgsz=core["imgsz"],
                    epochs=epochs,
                    batch=core["batch"],
                    workers=int(args.workers),
                    optimizer=core["optimizer"],     # 'auto' is fine for v8
                    lr0=core["lr0"],
                    lrf=core["lrf"],
                    wd=core["weight_decay"],
                    conf=conf,
                    iou=iou,
                    project=args.project,
                    fold_name=fold_name,
                )
            elif model == "yolov3":


                map50, rmse = yolov3_train_eval(
                    base_weights=base_cfg["model"].get("weights"),
                    data_yaml=fold_yaml,
                    imgsz=core["imgsz"],
                    epochs=epochs,
                    batch=core["batch"],
                    workers=args.workers,
                    optimizer=core["optimizer"],
                    lr0=core["lr0"],
                    lrf=core["lrf"],
                    wd=core["weight_decay"],
                    conf=conf,
                    iou=iou,
                    project=args.project,
                    fold_name=fold_name,
                    device="0",  # or "cpu"
                )

            score = composite_score(map50, rmse, gtc)
            per_fold_scores.append(score)
            per_fold_map.append(map50)
            per_fold_rmse.append(rmse)

            if wb:
                wandb.log({f"fold{k}/map50": map50, f"fold{k}/rmse": rmse, f"fold{k}/score": score})

            # report intermediate result to enable pruning
            trial.report(float(np.mean(per_fold_scores)), step=k)
            if trial.should_prune():
                if wb:
                    wandb.run.summary["trial/pruned_at_fold"] = k
                raise optuna.TrialPruned(f"Pruned at fold {k}")

        final_score = float(np.mean(per_fold_scores))
        if wb:
            wandb.run.summary["score/final"] = final_score
            wandb.run.summary["map50/avg"] = float(np.mean(per_fold_map)) if per_fold_map else float("nan")
            wandb.run.summary["rmse/avg"] = float(np.mean(per_fold_rmse)) if per_fold_rmse else float("nan")
            wandb.run.summary["params"] = trial.params

        logger.info(

            f"\n\n\n[HPO] Trial {trial.number} score={final_score:.4f} "
            f"(map50={np.mean(per_fold_map):.4f}, rmse={np.mean(per_fold_rmse):.4f})\n\n\n"
        )
        return final_score

    return objective
