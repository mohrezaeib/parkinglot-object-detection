# src/train/kfold.py
from pathlib import Path
from sklearn.model_selection import KFold
import glob

def _list_images(img_dir: Path):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
    return sorted([p for p in glob.glob(str(img_dir / "*")) if p.endswith(exts)])

def _write_list(paths, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in paths:
            f.write(str(Path(p).resolve()) + "\n")

def make_kfold_splits(dataset_root: Path, kfolds: int, seed: int = 42) -> Path:
    """Create TXT lists: splits/fold{k}_train.txt & splits/fold{k}_val.txt."""
    img_train_dir = dataset_root / "images/train"
    images = _list_images(img_train_dir)
    if len(images) < kfolds:
        raise ValueError(f"Not enough training images ({len(images)}) for {kfolds} folds.")

    out_dir = dataset_root / "splits"
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(kf.split(images)):
        val_set = [images[i] for i in val_idx]
        train_set = [p for i, p in enumerate(images) if i not in val_idx]
        _write_list(train_set, out_dir / f"fold{fold}_train.txt")
        _write_list(val_set,   out_dir / f"fold{fold}_val.txt")
    return out_dir

def ensure_kfold_splits(dataset_root: Path, k: int, seed: int) -> Path:
    splits_dir = dataset_root / "splits"
    need = any(not (splits_dir / f"fold{f}_train.txt").exists() or not (splits_dir / f"fold{f}_val.txt").exists() for f in range(k))
    if need:
        return make_kfold_splits(dataset_root, k, seed)
    return splits_dir
