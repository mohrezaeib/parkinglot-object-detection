import shutil, subprocess
from pathlib import Path

class YOLOUltralytics:
    def __init__(self, variant="v8"):
        self.variant = variant
        self.weights_path = None

    # YOLOv8 via ultralytics API
    def _train_v8(self, data_yaml, imgsz, epochs, batch, project, fold_name, lr0=None, lrf=None, wd=None, **kwargs):
        from ultralytics import YOLO
        model_name = kwargs.get("model_name", "yolov8n.pt")
        model = YOLO(model_name)
        results = model.train(
            data=data_yaml, imgsz=imgsz, epochs=epochs, batch=batch,
            project=project, name=fold_name, workers=kwargs.get("workers", 4),
            optimizer=kwargs.get("optimizer", "auto"),
            lr0=lr0, lrf=lrf, weight_decay=wd,
            pretrained=True, seed=kwargs.get("seed", 42),
            device=kwargs.get("device", 0), plots=True
        )
        self.weights_path = str(Path(results.save_dir) / "best.pt")
        return self.weights_path

    def _val_v8(self, weights, data_yaml, imgsz, split="val", conf=0.25, iou=0.5):
        from ultralytics import YOLO
        model = YOLO(weights)
        m = model.val(data=data_yaml, imgsz=imgsz, split=split, conf=conf, iou=iou, save_txt=False)
        return m

    def _predict_v8(self, weights, source, imgsz=640, conf=0.25, iou=0.5, save_vis=False, outdir="runs/predict"):
        from ultralytics import YOLO
        model = YOLO(weights)
        res = model.predict(source=source, imgsz=imgsz, conf=conf, iou=iou, save=save_vis, project=outdir)
        return res

    # YOLOv5/YOLOv3 via external repos
    def _train_external(self, repo_path, data_yaml, imgsz, epochs, batch, project, fold_name, **kwargs):
        py = shutil.which("python") or "python"
        cmd = [
            py, "train.py",
            "--img", str(imgsz),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--data", str(data_yaml),
            "--project", str(project),
            "--name", str(fold_name),
            "--exist-ok",
            "--single-cls"
        ]
        if kwargs.get("weights"):
            cmd += ["--weights", kwargs["weights"]]
        return subprocess.check_call(cmd, cwd=repo_path)

    def _predict_external(self, repo_path, weights, source, imgsz=640, conf=0.25, iou=0.5, save_vis=False, outdir="runs/predict"):
        py = shutil.which("python") or "python"
        cmd = [
            py, "detect.py",
            "--img", str(imgsz),
            "--conf", str(conf),
            "--iou", str(iou),
            "--source", str(source),
            "--project", str(outdir),
            "--exist-ok"
        ]
        if weights: cmd += ["--weights", weights]
        if save_vis: cmd += ["--save-txt", "--save-conf"]
        return subprocess.check_call(cmd, cwd=repo_path)

    def train(self, **kwargs):
        if self.variant == "v8":
            return self._train_v8(**kwargs)
        elif self.variant == "v5":
            repo = kwargs.get("repo", "external/yolov5")
            return self._train_external(repo, **kwargs)
        elif self.variant == "v3":
            repo = kwargs.get("repo", "external/yolov3")
            return self._train_external(repo, **kwargs)
        else:
            raise ValueError(self.variant)

    def validate(self, **kwargs):
        if self.variant == "v8":
            return self._val_v8(**kwargs)
        else:
            raise NotImplementedError("Use repo-native val script for v5/v3 or evaluate.py to compute metrics.")

    def predict(self, **kwargs):
        if self.variant == "v8":
            return self._predict_v8(**kwargs)
        else:
            repo = kwargs.get("repo", f"external/yolo{self.variant}")
            return self._predict_external(repo, **kwargs)
