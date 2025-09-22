# tools/make_kfold.py
import argparse
from pathlib import Path
from sklearn.model_selection import KFold
import glob, os

def list_images(img_dir):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
    return sorted([p for p in glob.glob(str(Path(img_dir) / "*")) if p.endswith(exts)])

def write_list(paths, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + "\n")

def make_kfold_splits(dataset_root: Path, kfolds: int, seed: int = 42):
    img_train_dir = dataset_root / "images/train"
    images = list_images(img_train_dir)
    if len(images) < kfolds:
        raise ValueError(f"Not enough training images ({len(images)}) for {kfolds} folds.")

    out_dir = dataset_root / "splits"
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(kf.split(images)):
        val_set = [images[i] for i in val_idx]
        train_set = [p for i, p in enumerate(images) if i not in val_idx]
        write_list(train_set, out_dir / f"fold{fold}_train.txt")
        write_list(val_set,   out_dir / f"fold{fold}_val.txt")
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="datasets/yolo", help="Root containing images/ and labels/")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = make_kfold_splits(Path(args.data_root), args.folds, args.seed)
    print(f"Saved split files to {out}")

if __name__ == "__main__":
    main()
