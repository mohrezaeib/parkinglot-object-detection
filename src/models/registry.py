from .yolo_ultralytics import YOLOUltralytics
from .torchvision_frcnn import TorchvisionFasterRCNN

def get_model_adapter(name: str):
    name = name.lower()
    if name == "yolov8":
        return YOLOUltralytics(variant="v8")
    if name == "yolov5":
        return YOLOUltralytics(variant="v5")
    if name == "yolov3":
        return YOLOUltralytics(variant="v3")
    if name == "fasterrcnn":
        return TorchvisionFasterRCNN()
    raise ValueError(f"Unknown model backend: {name}")
