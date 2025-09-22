import argparse
from ultralytics import YOLO

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov8")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    return ap.parse_args()

def main():
    a = parse()
    if a.model != "yolov8":
        print("Predict helper focuses on YOLOv8. Use external scripts for v5/v3.")
    YOLO(a.weights).predict(source=a.source, imgsz=a.imgsz, conf=a.conf, iou=a.iou, save=True)

if __name__ == "__main__":
    main()
