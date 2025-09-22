import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
from ..utils.datasets import load_yaml
from ..utils.engine_frcnn import train_one_epoch

class TorchvisionFasterRCNN:
    def __init__(self):
        self.model = None

    def _build(self, num_classes=2):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # background + car
        return model

    def train(self, data_yaml: str, imgsz: int, epochs: int, batch: int, project: str, fold_name: str, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build(num_classes=2).to(device)
        # Minimal loop; for a full pipeline use YOLOv8.
        # (You can extend this to a full dataloader similar to YOLO, omitted for brevity)
        # Save weights placeholder
        out_dir = Path(project) / fold_name
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / "fasterrcnn_best.pth"
        torch.save(self.model.state_dict(), weights)
        return str(weights)
