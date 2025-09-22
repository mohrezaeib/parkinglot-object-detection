# train.py (unchanged and slim)
from pathlib import Path
import wandb
from loguru import logger
from src.train.cli import parse_args
from src.utils.datasets import load_yaml
from src.train.kfold import ensure_kfold_splits
from src.train.fold_yaml import create_fold_yaml
from src.train.unified_trainer import run_one_fold  

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    exp = cfg["experiment"]
    if args.model: cfg["model"]["name"] = args.model
    if args.epochs: exp["epochs"] = args.epochs
    if args.imgsz: exp["imgsz"] = args.imgsz
    if args.batch: exp["batch"] = args.batch
    if args.seed is not None: exp["seed"] = args.seed

    base_yaml_path = cfg["dataset"]["yaml"]
    dataset_root = Path(load_yaml(base_yaml_path)["path"])
    kfolds = int(cfg["dataset"].get("folds", 5))
    splits_dir = ensure_kfold_splits(dataset_root, kfolds, exp.get("seed", 42))

    wb_proj = args.wandb_project or cfg["wandb"]["project"]
    wb_entity = args.wandb_entity or cfg["wandb"]["entity"]
    wb_run = args.wandb_run_name or cfg["wandb"]["run_name"]

    io_cfg = {
        "project": args.project,
        "workers": args.workers,
        "weights": args.weights,
        "kfolds": kfolds,
        "wandb_project": wb_proj,
        "wandb_entity": wb_entity,
    }

    model_name = cfg["model"]["name"]
    for k in range(kfolds):
        run = wandb.init(project=wb_proj, entity=wb_entity,
                         name=wb_run or f"{model_name}_fold{k}", reinit=True)
        fold_train_txt = splits_dir / f"fold{k}_train.txt"
        fold_val_txt   = splits_dir / f"fold{k}_val.txt"
        fold_yaml_path = create_fold_yaml(base_yaml_path, fold_train_txt, fold_val_txt)
        logger.info(f"Training {model_name} fold {k+1}/{kfolds}")

        run_one_fold(model_name=model_name,
                     data_yaml=fold_yaml_path,
                     exp=exp,
                     model_cfg=cfg["model"],
                     io_cfg=io_cfg,
                     fold_index=k)
        wandb.finish()

if __name__ == "__main__":
    main()
