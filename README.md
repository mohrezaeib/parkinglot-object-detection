# ParkingLot-OD: Object Detection in Parking Lot Drone Images (CARPK & PUCPR+)

A clean, modular PyTorch project for single-class **car** detection on aerial parking lots.
Backends: **YOLOv8**, **YOLOv5**, **YOLOv3**, plus a **Faster R-CNN** baseline.
Includes **K-fold validation**, **Optuna hyperparameter search**, **Weights & Biases** logging, conversion utilities for **CARPK** and **PUCPR+**, and evaluation scripts (mAP/IoU + **car-count MAE/RMSE**).

---

## Features
- 🔁 Pluggable backends: `yolov8`, `yolov5`, `yolov3`, `fasterrcnn`
- ⚙️ Single config (`configs/config.yaml`) with CLI overrides
- 🧪 K-fold validation over training set
- 🔍 Hyperparameter tuning with Optuna
- 📊 W&B logging (train/val losses, metrics, example images)
- 🧰 Dataset prep for **CARPK** and **PUCPR+** → YOLO format (single class `car`)
- ✅ Evaluation: mAP@0.5 (Ultralytics), IoU stats, **car count** MAE/RMSE

---

## Setup
```bash
conda create -n parkinglot-od python=3.10 -y
conda activate parkinglot-od

# Install PyTorch for your CUDA: https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Optional external repos for YOLOv5/YOLOv3 wrappers
git clone https://github.com/ultralytics/yolov5 external/yolov5
git clone https://github.com/ultralytics/yolov3 external/yolov3
```
`ultralytics` (YOLOv8) is installed via pip.

---

## Data layout (input devkits)
```
datasets/
 ├─ CARPK_devkit/
 │   └─ data/
 │       ├─ Annotations/   # .txt lines: x1 y1 x2 y2 class_id
 │       ├─ Images/        # .png
 │       └─ ImageSets/     # file stems (no extension)
 │           ├─ train.txt
 │           └─ test.txt
 └─ PUCPR+_devkit/
     └─ data/
         ├─ Annotations/   # .txt lines: x1 y1 x2 y2 class_id
         ├─ Images/        # .jpg
         └─ ImageSets/
             ├─ train.txt
             └─ test.txt
```

### Convert to YOLO format & create dataset YAML
```bash
python tools/prepare_datasets.py   --carpk-root datasets/CARPK_devkit/data   --pucpr-root datasets/PUCPR+_devkit/data   --out datasets/yolo
```
Creates:
```
datasets/yolo/
 ├─ images/{train,val,test}/
 ├─ labels/{train,val,test}/
 └─ parkinglot.yaml
```
> We map any class IDs in source annotations to a single class `0: car`.

---

## Train (K-fold) — YOLOv8 example
```bash
python train.py --model yolov8 --folds 5 --epochs 150 --imgsz 640   --batch 16 --project runs/yolov8_kfold --wandb_project ParkingLotOD
```
Notes:
- For YOLOv8 we use the default `ultralytics` training which expects a `val` split.
- If your `val` split is empty, we still run K-fold by training on the `train` split and validating per-fold using a generated list. For simplicity, this scaffold trains with `val` if present; otherwise metrics are reported from training/validation hooks as available.

### Hyperparameter search (Optuna)
```bash
python hparam_search.py --model yolov8 --trials 25 --epochs 80   --project runs/hparam_yolov8 --wandb_project ParkingLotOD
```

### Evaluate on test (mAP + **car-count MAE/RMSE**)
```bash
python evaluate.py --model yolov8 --weights runs/yolov8_kfold/fold0/best.pt   --data datasets/yolo/parkinglot.yaml --split test --imgsz 640 --conf 0.25 --iou 0.5 --save_vis
```

### Predict
```bash
python predict.py --model yolov8 --weights path/to/best.pt --source path/to/image_or_dir
```

---

## Config
See `configs/config.yaml` for defaults (img size, epochs, batch, lr/weight decay, folds, etc.).

## W&B
- Export `WANDB_API_KEY` or `wandb login`.
- CLI flags: `--wandb_project`, `--wandb_entity`, `--wandb_run_name`.

## License
MIT for this scaffold. Respect dataset & Ultralytics licenses.
