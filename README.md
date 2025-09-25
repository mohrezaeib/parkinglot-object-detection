Here’s a minimal README refresh that matches the current code (legacy eval via `val.py`, modular runners, and the CLIs you’ve been using). I kept changes small and focused.

---

# ParkingLot-OD: Object Detection in Parking Lot Drone Images (CARPK & PUCPR+)

A clean, modular PyTorch project for single-class **car** detection on aerial parking lots.
Backends: **YOLOv8**, **YOLOv5**, **YOLOv3**, plus a **Faster R-CNN** baseline.
Includes **K-fold validation**, **Optuna hyperparameter search**, **Weights & Biases** logging, conversion utilities for **CARPK** and **PUCPR+**, and evaluation scripts (mAP/IoU + **car-count MAE/RMSE**).

---

## Features

* 🔁 Pluggable backends: `yolov8`, `yolov5`, `yolov3`, `fasterrcnn`
* ⚙️ Single config (`configs/*.yaml`) with CLI overrides
* 🧪 K-fold validation over training set
* 🔍 Hyperparameter tuning with Optuna
* 📊 W\&B logging (train/val losses, metrics, example images)
* 🧰 Dataset prep for **CARPK** and **PUCPR+** → YOLO format (single class `car`)
* ✅ Evaluation: mAP\@0.5 and **car-count** MAE/RMSE

  * YOLOv8: native `ultralytics` API
  * YOLOv5/YOLOv3 (legacy): repo **`val.py`** + label files → count metrics

---

## Setup

```bash
conda create -n parkinglot-od python=3.10 -y
conda activate parkinglot-od

# Install PyTorch for your CUDA: https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Optional external repos for legacy backends
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
 │       ├─ Annotations/   # .txt: x1 y1 x2 y2 class_id
 │       ├─ Images/        # .png
 │       └─ ImageSets/
 │           ├─ train.txt
 │           └─ test.txt
 └─ PUCPR+_devkit/
     └─ data/
         ├─ Annotations/
         ├─ Images/        # .jpg
         └─ ImageSets/
             ├─ train.txt
             └─ test.txt
```

### Convert to YOLO format & dataset YAML

```bash
python tools/prepare_datasets.py \
  --carpk-root datasets/CARPK_devkit/data \
  --pucpr-root datasets/PUCPR+_devkit/data \
  --out datasets/yolo
```

Produces:

```
datasets/yolo/
 ├─ images/{train,val,test}/
 ├─ labels/{train,val,test}/
 └─ parkinglot.yaml
```

> All classes are mapped to a single class `0: car`.

---

## Train (K-fold) — YOLOv8 example

```bash
python train.py \
  --model yolov8 --folds 5 --epochs 150 --imgsz 640 --batch 16 \
  --project runs/yolov8_kfold --wandb_project ParkingLotOD
```
or

```bash
python train.py --config configs/yolov8.yaml
```
## Evaluate on test (mAP + **count MAE/RMSE**)

Unified CLI:

```bash
# YOLOv8
python evaluate.py \
  --model yolov8 \
  --weights runs/yolov8_kfold/fold0/best.pt \
  --data datasets/yolo/parkinglot.yaml --split test --imgsz 640 --conf 0.25 --iou 0.5 --save_vis

# YOLOv5
python evaluate.py \
  --model yolov5 \
  --weights external/yolov5/runs/train/fold0/weights/best.pt \
  --data datasets/yolo/parkinglot.yaml --split test --imgsz 640 --conf 0.25 --iou 0.5

# YOLOv3
python evaluate.py \
  --model yolov3 \
  --weights external/yolov3/runs/train/fold0/weights/best.pt \
  --data datasets/yolo/parkinglot.yaml --split test --imgsz 640 --conf 0.25 --iou 0.5
```

* **Legacy (v5/v3)** eval uses the repo’s **`val.py`** and reads its saved `labels/*.txt` to compute counts.

---

## Hyperparameter search (Optuna)

```bash
# YOLOv8
python hparam_search.py \
  --model yolov8 --config configs/yolov8.yaml \
  --trials 25 --epochs 80 --project runs/hparam/yolov8 --workers 4

# YOLOv3
python hparam_search.py \
  --model yolov3 --config configs/yolov3.yaml \
  --trials 10 --epochs 50 --project runs/hparam/yolov3 --workers 4
```

Notes:

* **YOLOv3** doesn’t accept `--lr0/--lrf/--weight-decay` on CLI. We write a temporary **hyp YAML** and pass it via `--hyp` internally (handled by the runner).
* W\&B trial runs are named `HPO-<model>-t<trial>` and log per-fold metrics + composite score.
* **Login to Weight and Biases** before Training or HPO 
---

## Configs

* Use the per-backend defaults in `configs/yolov8.yaml`, `configs/yolov5.yaml`, `configs/yolov3.yaml`.
* Common keys:

  * `experiment`: `imgsz`, `epochs`, `batch`, `conf_thres`, `iou_thres`, `seed`, …
  * `dataset`: path to `datasets/yolo/parkinglot.yaml`, folds, fold\_index.
  * `wandb`: `project`, `entity`, `run_name`.
  * `model`: `name`, `weights`, `optimizer`, `lr0`, `lrf`, `weight_decay`.

---

## W\&B

* `wandb login` or set `WANDB_API_KEY`.
* CLI flags: `--wandb_project`, `--wandb_entity`, `--wandb_run_name`.
* Internal legacy runners disable the repo’s own W\&B to avoid duplicate streams; metrics are logged from our code.

---

## Tips

* **GPU selection**: set `CUDA_VISIBLE_DEVICES=0` (or pass `--device 0` where supported).
* **Legacy repos**: this code expects the stock `train.py`/`val.py` entrypoints in `external/yolov5` and `external/yolov3`.

---

## License

MIT for this scaffold. Please respect dataset and Ultralytics licenses.
