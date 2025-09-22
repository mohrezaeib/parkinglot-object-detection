# hparam_search.py
from __future__ import annotations
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

from src.hpo.cli import parse_args
from src.utils.datasets import load_yaml
from src.hpo.objective import make_objective

def main():
    args = parse_args()
    base_cfg = load_yaml(args.config)

    sampler = TPESampler(
        multivariate=True, group=True, n_startup_trials=5,
        seed=base_cfg["experiment"].get("seed", 42)
    )
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                study_name=f"HPO-{args.model}")

    timeout_sec = args.timeout_min * 60 if args.timeout_min else None
    study.optimize(make_objective(base_cfg, args), n_trials=args.trials,
                   timeout=timeout_sec, show_progress_bar=True)
    try:
        print("\nBest params:", study.best_params)
        print("Best score:", study.best_value)
    except ValueError:
        # No completed trials — likely all pruned
        print("\nNo completed trials (all pruned). "
            "Consider relaxing pruning or reducing epochs per trial.")


if __name__ == "__main__":
    main()
