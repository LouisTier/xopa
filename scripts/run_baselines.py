"""Standalone selection baselines: Boruta plus one-shot rankers at a fixed K.

Example:
    uv run python scripts/run_baselines.py --data /abs/path/dataset.csv --k 70
"""

import argparse
import json
from functools import partial

from xopa.data import (
    CANDCE_2026_DATASET_CONTRACT,
    load_dataset_bundle,
    make_splits,
    validate_dataset_identity,
)
from xopa.models import evaluate, fit_model, make_model
from xopa.selection.boruta import run_boruta
from xopa.selection.rankers import RANKERS
from xopa.selection.rffs import jaccard_overlap
from xopa.utils import (
    load_config,
    timestamp_dir,
    write_common_artifacts,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the dataset file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--label", default="baselines")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Authors only: verify --data is the paper's dataset before running",
    )
    parser.add_argument("--backbone", default="xgboost")
    parser.add_argument("--k", type=int, default=70, help="Fixed selection size")
    parser.add_argument("--boruta-max-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["split"]["seed"]
    model_params = config["models"][args.backbone]
    early_stopping = config["models"]["early_stopping_rounds"]

    dataset = load_dataset_bundle(args.data, target=config["data"]["target"])
    if args.paper:
        validate_dataset_identity(dataset.identity, CANDCE_2026_DATASET_CONTRACT)
    splits = make_splits(
        dataset.x,
        dataset.y,
        tuple(config["split"]["ratios"]),
        seed,
        standardize_target=bool(config["data"].get("standardize_target", False)),
    )
    factory = partial(make_model, args.backbone, model_params, seed)

    boruta_cfg = dict(config["boruta"])
    if args.boruta_max_iter:
        boruta_cfg["max_iter"] = args.boruta_max_iter
    boruta = run_boruta(
        splits.X_train,
        splits.y_train,
        factory,
        max_iter=boruta_cfg["max_iter"],
        alpha=boruta_cfg["alpha"],
        perc=boruta_cfg["perc"],
        two_step=boruta_cfg["two_step"],
        seed=seed,
    )

    sets: dict[str, list[str]] = {"boruta_confirmed": boruta.confirmed}
    rankers = (
        ["rfe", "mrmr", "lasso", "tree"]
        if args.k % 10 == 0
        else ["mrmr", "lasso", "tree"]
    )
    for name in rankers:
        sets[f"{name}_k{args.k}"] = RANKERS[name](
            splits.X_train,
            splits.y_train,
            factory,
            k_max=args.k,
            seed=seed,
            checkpoints=[args.k],
        )

    val_metrics = {}
    for label, features in sets.items():
        if not features:
            val_metrics[label] = None
            continue
        model = fit_model(
            make_model(args.backbone, model_params, seed),
            splits.X_train[features],
            splits.y_train,
            splits.X_val[features],
            splits.y_val,
            early_stopping_rounds=early_stopping,
        )
        val_metrics[label] = evaluate(model, splits.X_val[features], splits.y_val)

    out = timestamp_dir(args.out, f"{args.label}_{args.backbone}")
    (out / "selected_sets.json").write_text(json.dumps(sets, indent=2))
    (out / "boruta_decision.json").write_text(
        json.dumps(
            {
                "confirmed": boruta.confirmed,
                "tentative": boruta.tentative,
                "tentative_supported": boruta.tentative_supported,
                "rejected": boruta.rejected,
                "ranking": boruta.ranking,
                "n_iter": boruta.n_iter,
            },
            indent=2,
        )
    )
    (out / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    boruta.importance_history.to_csv(out / "boruta_importance_history.csv")
    boruta.decision_history.to_csv(out / "boruta_decision_history.csv", index=False)
    jaccard_overlap(sets).to_csv(out / "overlap_jaccard.csv")

    write_common_artifacts(out, dataset, splits)
    write_manifest(
        out,
        config,
        {"cli": vars(args), "boruta_params": boruta.params},
        stage="baselines",
        run_kind="paper" if args.paper else "user",
    )
    print(f"Baselines done: boruta confirmed {len(boruta.confirmed)} -> {out}")


if __name__ == "__main__":
    main()
