"""Run the full RFFS campaign (FFS grid + SFS refinement) for one backbone.

Example:
    uv run python scripts/run_rffs.py --data /abs/path/dataset.csv \
        --backbone xgboost
"""

import argparse
import json
import logging

import pandas as pd

from xopa.data import (
    CANDCE_2026_DATASET_CONTRACT,
    exclude_categories,
    feature_category,
    load_dataset_bundle,
    make_splits,
    validate_dataset_identity,
)
from xopa.selection.probes import ProbeConfig, is_probe
from xopa.selection.rffs import run_rffs
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
    parser.add_argument("--label", default="rffs")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Authors only: verify --data is the paper's dataset before running",
    )
    parser.add_argument("--backbone", default="xgboost")
    parser.add_argument("--operators", nargs="+", default=None)
    parser.add_argument("--grid", nargs="+", type=int, default=None)
    parser.add_argument("--tau", default=None, help='Float or "auto"')
    parser.add_argument("--q", type=int, default=None, help="FFS iterations Q")
    parser.add_argument("--sfs-repeats", type=int, default=None)
    parser.add_argument("--shap-subsample", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for the fit loops; use for the single-threaded "
        "MLP backbone, keep 1 for tree backbones (results are identical)",
    )
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument(
        "--exclude-categories",
        nargs="+",
        default=None,
        help="Feature categories to drop after identity validation (e.g. PRS)",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def selection_frequency_table(report, q_iterations: int, tau_setting) -> pd.DataFrame:
    """Tidy per-(operator, K, feature) selection-frequency table."""
    rows = []
    for operator, result in report.ffs.items():
        combined = pd.concat([result.counts, result.probe_counts])
        for k in result.counts.columns:
            tau_k = result.tau_used[k]
            for order, (feature, count) in enumerate(combined[k].items()):
                probe = is_probe(feature)
                try:
                    category = "probe" if probe else feature_category(feature)
                except ValueError:
                    category = "other"
                rows.append(
                    {
                        "operator": operator,
                        "K": k,
                        "feature": feature,
                        "feature_order": order,
                        "category": category,
                        "is_probe": probe,
                        "count": int(count),
                        "frequency": float(count) / q_iterations,
                        "Q": q_iterations,
                        "tau": tau_k,
                        "tau_source": "auto" if tau_setting == "auto" else "fixed",
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    ffs_cfg, sfs_cfg = config["ffs"], config["sfs"]
    operators = args.operators or ffs_cfg["operators"]
    grid = args.grid or ffs_cfg["grid"]
    tau = args.tau if args.tau is not None else ffs_cfg["tau"]
    if tau != "auto":
        tau = float(tau)
    q_iterations = args.q or ffs_cfg["q_iterations"]
    seed = args.seed if args.seed is not None else config["split"]["seed"]
    probes = None if args.no_probes else ProbeConfig(**ffs_cfg["probes"])

    dataset = load_dataset_bundle(args.data, target=config["data"]["target"])
    if args.paper:
        validate_dataset_identity(dataset.identity, CANDCE_2026_DATASET_CONTRACT)
    x, y = dataset.x, dataset.y
    excluded: list[str] = []
    if args.exclude_categories:
        x, excluded = exclude_categories(x, args.exclude_categories)
        logging.info("excluded %d features; %d remain", len(excluded), x.shape[1])
        oversized = [k for k in grid if k > x.shape[1]]
        if oversized:
            raise ValueError(f"grid values {oversized} exceed remaining features")
    splits = make_splits(
        x,
        y,
        tuple(config["split"]["ratios"]),
        seed,
        standardize_target=bool(config["data"].get("standardize_target", False)),
    )

    report = run_rffs(
        splits,
        operators,
        grid,
        backbone=args.backbone,
        model_params=config["models"][args.backbone],
        q_iterations=q_iterations,
        rho=ffs_cfg["rho"],
        tau=tau,
        probes=probes,
        seed=seed,
        rfe_step=ffs_cfg["rfe_step"],
        sfs_n_repeats=args.sfs_repeats or sfs_cfg["n_repeats"],
        sfs_epsilon=sfs_cfg["epsilon"],
        shap_subsample=args.shap_subsample or sfs_cfg["shap_subsample"],
        early_stopping_rounds=config["models"]["early_stopping_rounds"],
        n_jobs=args.n_jobs,
    )

    out = timestamp_dir(args.out, f"{args.label}_{args.backbone}")
    rankings = []
    for operator, result in report.ffs.items():
        result.grid_eval.to_csv(out / f"grid_eval_{operator}.csv")
        result.counts.to_csv(out / f"counts_{operator}.csv")
        result.probe_counts.to_csv(out / f"probe_counts_{operator}.csv")
        result.probe_ranks.to_csv(out / f"probe_ranks_{operator}.csv")
        (out / f"survivors_{operator}.json").write_text(
            json.dumps(result.survivors, indent=2)
        )
        block = result.rankings
        if "operator" not in block.columns:
            block = block.assign(operator=operator)
        rankings.append(block)
    pd.concat(rankings, ignore_index=True).to_csv(out / "ffs_rankings.csv", index=False)
    selection_frequency_table(report, q_iterations, tau).to_csv(
        out / "selection_frequency.csv", index=False
    )
    report.sfs.curve.to_csv(out / "sfs_curve.csv")
    report.sfs.importance.rename("mean_abs_shap").to_csv(out / "sfs_importance.csv")
    report.category_table.to_csv(out / "category_summary.csv")
    report.overlap.to_csv(out / "overlap_jaccard.csv")
    report.test_predictions.to_csv(out / "rffs_test_predictions.csv", index=False)
    winner_operator, winner_k = report.winner
    summary = {
        "winner": {"operator": winner_operator, "K": winner_k},
        "winner_rule": report.params["ffs_winner_rule"],
        "d_plus": report.sfs.d_plus,
        "d_parsimonious": report.sfs.d_parsimonious,
        "probe_floor": report.sfs.probe_floor,
        "tau_used": {
            str(k): v for k, v in report.ffs[winner_operator].tau_used.items()
        },
        "tau_source": "auto" if tau == "auto" else "fixed",
        "final_features": report.final_features,
        "parsimonious_features": report.parsimonious_features,
        "test_metrics": report.test_metrics,
        "excluded_categories": args.exclude_categories or [],
        "excluded_features": excluded,
    }
    (out / "rffs_summary.json").write_text(json.dumps(summary, indent=2))

    write_common_artifacts(out, dataset, splits)
    write_manifest(
        out,
        config,
        {"cli": vars(args), "rffs_params": report.params},
        stage="rffs",
        run_kind="paper" if args.paper else "user",
    )
    print(f"RFFS done: winner {report.winner}, d_plus={report.sfs.d_plus} -> {out}")


if __name__ == "__main__":
    main()
