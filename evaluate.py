# evaluate.py
import argparse
from src.eval.yolov8_eval import eval_yolov8
from src.eval.yolo_legacy_eval import eval_legacy

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="yolov8|yolov5|yolov3")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="datasets/yolo/parkinglot.yaml")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="ParkingLotOD")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    return ap.parse_args()

def main():
    a = parse()
    m = a.model.lower()
    if m == "yolov8":
        eval_yolov8(a.weights, a.data, a.imgsz, a.split, a.conf, a.iou, a.save_vis, a.wandb_project, a.wandb_run_name)
    elif m in ("yolov5", "yolov3"):
        eval_legacy(m, a.weights, a.data, a.imgsz, a.split, a.conf, a.iou, a.save_vis, a.wandb_project, a.wandb_run_name)
    else:
        raise ValueError(f"Unsupported model: {a.model}")

if __name__ == "__main__":
    main()
