import yaml
from pathlib import Path
import cv2
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def list_split_files(root, split):
    root = Path(root)
    img_dir = root / f"images/{split}"
    return sorted([p for p in img_dir.glob("*.*") if p.suffix.lower() in [".jpg",".png",".jpeg"]])

def yolo_label_path(image_path, root):
    root = Path(root)
    lbl = root.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    return lbl

def read_yolo_labels(txt_path):
    objs = []
    if not txt_path.exists():
        return objs
    with open(txt_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                objs.append((int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return objs

def build_transforms(imgsz, augment=True):
    tr = []
    if augment:
        tr += [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.Cutout(num_holes=4, max_h_size=int(imgsz*0.1), max_w_size=int(imgsz*0.1), fill_value=0, p=0.3),
        ]
    tr += [A.LongestMaxSize(max_size=imgsz), A.PadIfNeeded(imgsz, imgsz, border_mode=cv2.BORDER_CONSTANT)]
    return A.Compose(tr, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

class YOLOFileDataset(Dataset):
    def __init__(self, yaml_path, split="train", imgsz=640, augment=True):
        self.cfg = load_yaml(yaml_path)
        self.root = Path(self.cfg["path"])
        self.imgs = list_split_files(self.root, split)
        self.split = split
        self.imgsz = imgsz
        self.tfm = build_transforms(imgsz, augment=augment and split=="train")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        im = cv2.imread(str(p))[:, :, ::-1]
        H, W = im.shape[:2]
        lbl_path = yolo_label_path(p, self.root)
        objs = read_yolo_labels(lbl_path)
        class_labels = [o[0] for o in objs]
        bboxes = [o[1:] for o in objs]
        transformed = self.tfm(image=im, bboxes=bboxes, class_labels=class_labels)
        img = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['class_labels']
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        target = {"boxes_yolo": bboxes, "labels": labels, "size": (H,W), "path": str(p)}
        return torch.tensor(img), target
