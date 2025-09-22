# src/hpo/runners/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple
from src.utils.datasets import load_yaml
from src.train.kfold import ensure_kfold_splits
from src.train.fold_yaml import create_fold_yaml

def kfold_iter(data_yaml: str, folds_to_eval: int, seed: int) -> Iterator[Tuple[int, str]]:
    """
    Yields (fold_index, fold_yaml_path) for k folds using your existing split tools.
    """
    dataset_root = Path(load_yaml(data_yaml)["path"])
    splits_dir = ensure_kfold_splits(dataset_root, folds_to_eval, seed)
    for k in range(folds_to_eval):
        yield k, create_fold_yaml(
            data_yaml,
            splits_dir / f"fold{k}_train.txt",
            splits_dir / f"fold{k}_val.txt",
        )
