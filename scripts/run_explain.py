"""Full-dataset SHAP explainability: drift, stability, category summary.

Attributions are computed over ALL records in original order (with the model
frozen there is no leakage risk), which the record-index drift analysis needs.

Example:
    uv run python scripts/run_explain.py --data /abs/path/dataset.csv \
        --features-file results/<run>/rffs_summary.json --highlight 22500 28499
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
from xopa.explain.shap_analysis import (
    category_summary,
    drift_by_index,
    drift_by_output,
    explanation_diagnostics,
    global_importance,
    shap_matrix,
    top_k_persistence,
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
    parser.add_argument("--label", default="explain")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Authors only: verify --data is the paper's dataset before running",
    )
    parser.add_argument("--backbone", default="xgboost")
    parser.add_argument(
        "--features-file",
        default=None,
        help="JSON list of features or an rffs_summary.json (final_features used)",
    )
    parser.add_argument("--n-display", type=int, default=10)
    parser.add_argument(
        "--highlight",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Record range annotation carried into the artifacts",
    )
    parser.add_argument("--shap-subsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_features(path: str | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return payload["final_features"] if isinstance(payload, dict) else payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    explain_cfg = config["explain"]
    seed = args.seed if args.seed is not None else config["split"]["seed"]
    model_params = config["models"][args.backbone]

    dataset = load_dataset_bundle(args.data, target=config["data"]["target"])
    if args.paper:
        validate_dataset_identity(dataset.identity, CANDCE_2026_DATASET_CONTRACT)
    features = load_features(args.features_file) or list(dataset.x.columns)
    splits = make_splits(
        dataset.x[features],
        dataset.y,
        tuple(config["split"]["ratios"]),
        seed,
        standardize_target=bool(config["data"].get("standardize_target", False)),
    )

    model = fit_model(
        make_model(args.backbone, model_params, seed),
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        early_stopping_rounds=config["models"]["early_stopping_rounds"],
    )
    test_metrics = evaluate(model, splits.X_test, splits.y_test)

    labeled_blocks = (
        ("train", splits.X_train, splits.y_train),
        ("validation", splits.X_val, splits.y_val),
        ("calibration", splits.X_calib, splits.y_calib),
        ("test", splits.X_test, splits.y_test),
    )
    x_full = pd.concat([x for _, x, _ in labeled_blocks]).sort_index()
    y_full = pd.concat([y for _, _, y in labeled_blocks]).sort_index()
    split_of = pd.concat(
        [pd.Series(label, index=x.index) for label, x, _ in labeled_blocks]
    ).sort_index()
    if args.shap_subsample is not None and args.shap_subsample < len(x_full):
        keep = x_full.index[:: max(1, len(x_full) // args.shap_subsample)]
        x_full, y_full, split_of = (
            x_full.loc[keep],
            y_full.loc[keep],
            split_of.loc[keep],
        )

    shap_values = shap_matrix(model, x_full, args.backbone)
    predictions = np.asarray(model.predict(x_full), dtype=float)
    importance = global_importance(shap_values)
    n_show = args.n_display
    display = (
        list(importance.index[: n_show // 2 + n_show % 2])
        + list(importance.index[-(n_show // 2) :])
        if len(importance) > n_show
        else list(importance.index)
    )

    drift_idx = drift_by_index(shap_values, display, explain_cfg["group_size"])
    drift_out = drift_by_output(shap_values, display, y_full, explain_cfg["bin_width"])
    stability, monotonic = explanation_diagnostics(
        shap_values,
        y_full,
        group_size=explain_cfg["group_size"],
        bin_width=explain_cfg["bin_width"],
    )
    persistence = top_k_persistence(
        shap_values, list(importance.index), group_size=explain_cfg["group_size"]
    )
    categories = category_summary(features, importance)

    out = timestamp_dir(args.out, f"{args.label}_{args.backbone}")
    importance.rename("mean_abs_shap").to_csv(out / "global_importance.csv")
    drift_idx.to_csv(out / "drift_by_index.csv")
    drift_out.to_csv(out / "drift_by_output.csv")
    stability.to_csv(out / "rank_stability.csv")
    persistence.rename_axis("group").reset_index().to_csv(
        out / "top5_persistence.csv", index=False
    )
    monotonic.to_csv(out / "monotonicity.csv")
    categories.to_csv(out / "category_summary.csv")
    (out / "model_test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    np.savez_compressed(
        out / "shap_bundle.npz",
        shap_values=shap_values.to_numpy(dtype=np.float32),
        feature_values=x_full.to_numpy(dtype=np.float32),
        feature_names=np.asarray(list(x_full.columns)),
        row_ids=x_full.index.to_numpy(),
        y_true=y_full.to_numpy(dtype=float),
        predictions=predictions,
        residuals=y_full.to_numpy(dtype=float) - predictions,
        split_labels=split_of.to_numpy(),
    )

    write_common_artifacts(out, dataset, splits)
    write_manifest(
        out,
        config,
        {
            "cli": vars(args),
            "n_features": len(features),
            "n_records": len(x_full),
            "regime_marker": args.highlight,
        },
        stage="explain",
        run_kind="paper" if args.paper else "user",
    )
    median_stability = float(pd.DataFrame(stability).iloc[:, 0].median())
    print(
        f"Explain done: {len(x_full):,} records, "
        f"median rank stability {median_stability:.2f} -> {out}"
    )


if __name__ == "__main__":
    main()
