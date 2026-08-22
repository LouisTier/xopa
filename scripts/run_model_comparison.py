"""Compare the four backbones on one or more feature sets.

Every model trains on the training split (early stopping on validation where
supported); the held-out test split is the headline.

Example:
    uv run python scripts/run_model_comparison.py --data /abs/path/dataset.csv \
        --feature-sets results/<run>/rffs_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xopa.data import (
    CANDCE_2026_DATASET_CONTRACT,
    load_dataset_bundle,
    make_splits,
    validate_dataset_identity,
)
from xopa.models import evaluate, fit_model, make_model
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
    parser.add_argument("--label", default="comparison")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Authors only: verify --data is the paper's dataset before running",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["xgboost", "random_forest", "lightgbm", "mlp"],
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=None,
        help="JSON files (feature lists or rffs_summary.json); default: all features",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_sets(paths: list[str] | None, all_features: list[str]) -> dict[str, list[str]]:
    if not paths:
        return {"all_features": all_features}
    sets: dict[str, list[str]] = {}
    for raw in paths:
        path = Path(raw)
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            sets[f"{path.stem}_d_plus"] = payload["final_features"]
            sets[f"{path.stem}_d_parsimonious"] = payload["parsimonious_features"]
        else:
            sets[path.stem] = payload
    sets["all_features"] = all_features
    return sets


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["split"]["seed"]
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
    feature_sets = load_sets(args.feature_sets, list(dataset.x.columns))

    metric_rows, prediction_rows = [], []
    for set_name, features in feature_sets.items():
        for backbone in args.backbones:
            model = fit_model(
                make_model(backbone, config["models"][backbone], seed),
                splits.X_train[features],
                splits.y_train,
                splits.X_val[features],
                splits.y_val,
                early_stopping_rounds=early_stopping,
            )
            for split_name, x_s, y_s in (
                ("train", splits.X_train[features], splits.y_train),
                ("val", splits.X_val[features], splits.y_val),
                ("test", splits.X_test[features], splits.y_test),
            ):
                metrics = evaluate(model, x_s, y_s)
                prediction = np.asarray(model.predict(x_s), dtype=float)
                metric_rows.append(
                    {
                        "feature_set": set_name,
                        "n_features": len(features),
                        "backbone": backbone,
                        "split": split_name,
                        **metrics,
                    }
                )
                truth = y_s.to_numpy(dtype=float)
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "feature_set": set_name,
                            "n_features": len(features),
                            "backbone": backbone,
                            "split": split_name,
                            "model_seed": seed,
                            "row_id": x_s.index,
                            "y_true": truth,
                            "prediction": prediction,
                            "residual": truth - prediction,
                        }
                    )
                )
            print(f"{set_name} / {backbone}: done")

    table = pd.DataFrame(metric_rows)
    out = timestamp_dir(args.out, args.label)
    table.to_csv(out / "model_comparison.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        out / "model_predictions.csv", index=False
    )
    (out / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2))

    write_common_artifacts(out, dataset, splits)
    write_manifest(
        out,
        config,
        {"cli": vars(args)},
        stage="comparison",
        run_kind="paper" if args.paper else "user",
    )
    pivot = table[table["split"] == "test"].pivot_table(
        index="feature_set", columns="backbone", values="r2"
    )
    print(f"Comparison done -> {out}\n{pivot.round(4)}")


if __name__ == "__main__":
    main()
