# src/eval/yolo_legacy_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import os, shutil, subprocess, json
import wandb
from src.train.common import (
    split_source_from_yaml,    
    labels_dir_for_split,      
    list_image_stems,          
    calc_gt_counts,                 
    count_mae_mse_rmse,        
)

def _legacy_detect(repo: str, weights: str, source: str, imgsz: int, conf: float, iou: float, save_vis: bool) -> List[int]:
    py = shutil.which("python") or "python"
    out_name = "legacy_eval"
    cmd = [
        py, "detect.py",
        "--weights", str(weights),
        "--img", str(imgsz),
        "--conf", str(conf),
        "--iou", str(iou),
        "--source", str(source),
        "--project", "runs/legacy_eval",
        "--name", out_name,
        "--exist-ok",
        "--save-txt", "--save-conf"
    ]
    if save_vis: cmd += ["--save-crop"]
    subprocess.check_call(cmd, cwd=repo)

    labels_dir = Path(repo) / "runs" / "legacy_eval" / out_name / "labels"
    stems = list_image_stems(source, source.endswith(".txt"))
    pred_counts = []
    for s in stems:
        p = labels_dir / f"{s}.txt"
        n = 0
        if p.exists():
            with open(p, "r") as fh:
                n = sum(1 for _ in fh if _.strip())
        pred_counts.append(n)
    return pred_counts

def _legacy_map(repo: str, data: str, weights: str, imgsz: int, conf: float, iou: float) -> Optional[float]:
    """Best-effort mAP@0.5 via val.py/test.py + latest results.csv if present."""
    py = shutil.which("python") or "python"
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"  # keep a single W&B run (this script)

    for s in ["val.py", "test.py"]:
        if (Path(repo) / s).exists():
            try:
                subprocess.check_call([
                    py, s,
                    "--data", str(Path(data).resolve()),
                    "--weights", str(weights),
                    "--img", str(imgsz),
                    "--conf", str(conf),
                    "--iou", str(iou),
                    "--task", "val",
                    "--single-cls",
                ], cwd=repo, env=env)
            except subprocess.CalledProcessError:
                pass
            break

    try:
        import pandas as pd
        csvs = sorted(Path(repo).glob("runs/**/results.csv"))
        if not csvs: return None
        df = pd.read_csv(csvs[-1])
        if "metrics/mAP_0.5" in df.columns:
            return float(df.iloc[-1]["metrics/mAP_0.5"])
        if "map50" in df.columns:
            return float(df.iloc[-1]["map50"])
    except Exception:
        return None
    return None

def eval_legacy(model: str, weights: str, data: str, imgsz: int, split: str, conf: float, iou: float,
                save_vis: bool, wb_project: str, wb_name: Optional[str]) -> dict:
    repo = "external/yolov5" if model == "yolov5" else "external/yolov3"
    wandb.init(project=wb_project, name=wb_name or f"eval-{Path(weights).stem}", reinit=True)

    source, is_txt = split_source_from_yaml(data, split)
    map50 = _legacy_map(repo, data, weights, imgsz, conf, iou)
    pred_counts = _legacy_detect(repo, weights, source, imgsz, conf, iou, save_vis)

    labels_dir = labels_dir_for_split(data, split, is_kfold_val_txt=is_txt)
    stems = list_image_stems(source, is_txt)
    gt = calc_gt_counts(labels_dir, stems)

    n = min(len(gt), len(pred_counts))
    mae, mse, rmse = count_mae_mse_rmse(gt[:n], pred_counts[:n])

    wandb.log({
        "eval/map50": map50 if map50 is not None else float("nan"),
        "eval/count_mae": mae,
        "eval/count_mse": mse,
        "eval/count_rmse": rmse,
        "eval/num_images": n,
        "eval/conf": conf,
        "eval/iou": iou,
        "eval/split": split,
    })
    out = {"map50": (map50 if map50 is not None else None),
           "MAE": mae, "MSE": mse, "RMSE": rmse, "N": n}
    print(json.dumps(out, indent=2))
    wandb.finish()
    return out
