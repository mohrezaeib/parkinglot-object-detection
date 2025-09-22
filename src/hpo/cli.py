# src/hpo/cli.py
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--model", type=str, default="yolov8", help="yolov8|yolov5|yolov3")
    ap.add_argument("--folds", type=int, default=2, help="K’ folds to average per trial (<= dataset.folds)")
    ap.add_argument("--epochs", type=int, default=60, help="epochs per trial (shorter for HPO)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--timeout_min", type=int, default=None)
    ap.add_argument("--project", type=str, default="runs/hparam")
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()
