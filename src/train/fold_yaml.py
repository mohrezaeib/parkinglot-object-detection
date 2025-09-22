# src/train/fold_yaml.py
from pathlib import Path
import tempfile
import yaml

def create_fold_yaml(base_yaml_path: str, fold_train_txt: Path, fold_val_txt: Path) -> str:
    """Build a temporary data YAML that points train/val to TXT file lists."""
    with open(base_yaml_path, "r") as f:
        base = yaml.safe_load(f)
    fold_yaml = {
        "path": str(Path(base["path"]).resolve()),
        "train": str(fold_train_txt.resolve()),
        "val": str(fold_val_txt.resolve()),
        "test": base.get("test", None),
        "names": base["names"],
    }
    tf = tempfile.NamedTemporaryFile("w", suffix=f"_fold.yaml", delete=False)
    yaml.safe_dump(fold_yaml, tf)
    tf.flush()
    return tf.name
