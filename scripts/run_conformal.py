"""Calibrate and evaluate m-SCP, m-CQR, and the ENS baseline.

The point model trains on the training split (early stopping on validation);
the calibration split is used exclusively here, for conformal calibration.

Example:
    uv run python scripts/run_conformal.py --data /abs/path/dataset.csv \
        --features-file results/<run>/rffs_summary.json --m 2200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xopa.conformal.cqr import fit_mcqr, fit_quantile_models, predict_mcqr
from xopa.conformal.ensemble import fit_ens, predict_ens
from xopa.conformal.metrics import mpiw, picp, running_average, wcc
from xopa.conformal.scp import fit_mscp, predict_mscp
from xopa.data import (
    CANDCE_2026_DATASET_CONTRACT,
    load_dataset_bundle,
    make_splits,
    validate_dataset_identity,
)
from xopa.models import fit_model, make_model
from xopa.utils import (
    load_config,
    spawn_seeds,
    timestamp_dir,
    write_common_artifacts,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the dataset file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--label", default="conformal")
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
    parser.add_argument("--methods", nargs="+", default=["scp", "cqr", "ens"])
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--slice",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Record range for regime-conditional PICP on the test split",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_features(path: str | None) -> list[str] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    return payload["final_features"] if isinstance(payload, dict) else payload


def conditional_block(
    y: np.ndarray, lb: np.ndarray, ub: np.ndarray, test_ids: np.ndarray, window
) -> tuple[dict, list[dict]]:
    """Overall + regime-window coverage in the recorded schema."""
    groups = {"overall": ("all test rows", np.ones(len(y), dtype=bool))}
    if window is not None:
        start, end = window
        inside = (test_ids >= start) & (test_ids <= end)
        groups[f"records_{start}_{end}"] = (
            f"original row_id in inclusive range [{start}, {end}]",
            inside,
        )
        groups["records_rest"] = (
            f"original row_id outside inclusive range [{start}, {end}]",
            ~inside,
        )
    payload, rows = {}, []
    for name, (definition, mask) in groups.items():
        entry = {
            "definition": definition,
            "n": int(mask.sum()),
            "picp": picp(y[mask], lb[mask], ub[mask]),
            "mpiw": mpiw(lb[mask], ub[mask]),
        }
        payload[name] = entry
        rows.append({"group": name, **entry})
    return payload, rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    conformal_cfg = config["conformal"]
    seed = args.seed if args.seed is not None else config["split"]["seed"]
    alpha = args.alpha if args.alpha is not None else conformal_cfg["alpha"]
    m = args.m if args.m is not None else conformal_cfg["m"]
    calib_fraction = conformal_cfg["calib_fraction"]
    model_params = config["models"][args.backbone]
    early_stopping = config["models"]["early_stopping_rounds"]

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
    scp_seed, cqr_seed, ens_seed = spawn_seeds(seed, 3)

    point_model = fit_model(
        make_model(args.backbone, model_params, seed),
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        early_stopping_rounds=early_stopping,
    )
    y_test = splits.y_test.to_numpy()
    test_ids = splits.y_test.index.to_numpy()
    point_pred = np.asarray(point_model.predict(splits.X_test), dtype=float)
    window = tuple(args.slice) if args.slice is not None else None

    out = timestamp_dir(args.out, f"{args.label}_{args.backbone}")
    metrics: dict[str, dict] = {}
    interval_rows: list[pd.DataFrame] = []
    convergence_rows: list[pd.DataFrame] = []
    checkpoint_rows: list[dict] = []
    conditional_rows: list[dict] = []
    resampling: dict[str, np.ndarray] = {}

    def record_intervals(method: str, lb: np.ndarray, ub: np.ndarray) -> None:
        covered = (y_test >= lb) & (y_test <= ub)
        interval_rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "row_position": np.arange(len(y_test)),
                    "row_id": test_ids,
                    "y_true": y_test,
                    "prediction": point_pred,
                    "lower": lb,
                    "upper": ub,
                    "covered": covered,
                    "width": ub - lb,
                }
            )
        )

    def record_msplit(method: str, result, base_lb, base_ub) -> None:
        """Convergence series, m-checkpoints, and resampling arrays."""
        n_calib = result.n_calib
        upper_bound = 1.0 - alpha + 1.0 / (n_calib + 1)
        convergence_rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "split_number": np.arange(1, len(result.thresholds) + 1),
                    "threshold": result.thresholds,
                    "eval_coverage": result.coverages,
                    "running_threshold": np.cumsum(result.thresholds)
                    / np.arange(1, len(result.thresholds) + 1),
                    "running_coverage": running_average(result.coverages),
                    "nominal_coverage": 1.0 - alpha,
                    "finite_sample_upper": upper_bound,
                    "n_calib": n_calib,
                }
            )
        )
        for checkpoint in (100, 500, len(result.thresholds)):
            if checkpoint > len(result.thresholds):
                continue
            q_c = float(np.mean(result.thresholds[:checkpoint]))
            lb_c, ub_c = base_lb - q_c, base_ub + q_c
            checkpoint_rows.append(
                {
                    "method": method,
                    "M": checkpoint,
                    "is_final": checkpoint == len(result.thresholds),
                    "q_hat": q_c,
                    "picp": picp(y_test, lb_c, ub_c),
                    "mpiw": mpiw(lb_c, ub_c),
                    "wcc": wcc(result.coverages[:checkpoint]),
                    "n_test": len(y_test),
                    "nominal_coverage": 1.0 - alpha,
                }
            )
        resampling[f"{method}_scores"] = result.scores
        resampling[f"{method}_calibration_masks"] = result.calibration_masks
        resampling[f"{method}_split_seeds"] = result.split_seeds
        resampling[f"{method}_row_ids"] = result.calibration_row_ids

    if "scp" in args.methods:
        result = fit_mscp(
            point_model,
            splits,
            alpha=alpha,
            m=m,
            calib_fraction=calib_fraction,
            seed=scp_seed,
        )
        lb, ub = predict_mscp(point_model, splits.X_test, result)
        record_intervals("scp", lb, ub)
        record_msplit("scp", result, point_pred, point_pred)
        conditional, rows = conditional_block(y_test, lb, ub, test_ids, window)
        conditional_rows += [{"method": "scp", **row} for row in rows]
        metrics["scp"] = {
            "picp": picp(y_test, lb, ub),
            "mpiw": mpiw(lb, ub),
            "wcc": wcc(result.coverages),
            "q_hat": result.q_hat,
            "conditional": conditional,
        }

    if "cqr" in args.methods:
        lo, hi = fit_quantile_models(
            splits,
            config["models"]["xgboost"],
            alpha=alpha,
            seed=seed,
            early_stopping_rounds=early_stopping,
        )
        result = fit_mcqr(
            lo,
            hi,
            splits,
            alpha=alpha,
            m=m,
            calib_fraction=calib_fraction,
            seed=cqr_seed,
        )
        lb, ub = predict_mcqr(lo, hi, splits.X_test, result)
        record_intervals("cqr", lb, ub)
        record_msplit(
            "cqr",
            result,
            np.minimum(lo.predict(splits.X_test), hi.predict(splits.X_test)),
            np.maximum(lo.predict(splits.X_test), hi.predict(splits.X_test)),
        )
        conditional, rows = conditional_block(y_test, lb, ub, test_ids, window)
        conditional_rows += [{"method": "cqr", **row} for row in rows]
        metrics["cqr"] = {
            "picp": picp(y_test, lb, ub),
            "mpiw": mpiw(lb, ub),
            "wcc": wcc(result.coverages),
            "q_hat": result.q_hat,
            **{k: v for k, v in result.params.items() if "cross" in k},
            "conditional": conditional,
        }

    if "ens" in args.methods:
        ens = fit_ens(
            splits,
            model_params,
            backbone=args.backbone,
            n_models=conformal_cfg["ens"]["n_models"],
            n_best=conformal_cfg["ens"]["n_best"],
            target_coverage=1.0 - alpha,
            seed=ens_seed,
            early_stopping_rounds=early_stopping,
        )
        lb, ub = predict_ens(ens, splits.X_test)
        record_intervals("ens", lb, ub)
        conditional, rows = conditional_block(y_test, lb, ub, test_ids, window)
        conditional_rows += [{"method": "ens", **row} for row in rows]
        metrics["ens"] = {
            "picp": picp(y_test, lb, ub),
            "mpiw": mpiw(lb, ub),
            "z": ens.z,
            "coverage_guarantee": "none",
            "conditional": conditional,
        }

    pd.concat(interval_rows, ignore_index=True).to_csv(
        out / "test_intervals.csv", index=False
    )
    if convergence_rows:
        pd.concat(convergence_rows, ignore_index=True).to_csv(
            out / "conformal_convergence.csv", index=False
        )
        pd.DataFrame(checkpoint_rows).to_csv(
            out / "conformal_checkpoints.csv", index=False
        )
        np.savez_compressed(out / "conformal_resampling.npz", **resampling)
    pd.DataFrame(conditional_rows).to_csv(out / "conditional_coverage.csv", index=False)
    (out / "uq_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    write_common_artifacts(out, dataset, splits)
    write_manifest(
        out,
        config,
        {"cli": vars(args), "n_features": len(features), "alpha": alpha, "m": m},
        stage="conformal",
        run_kind="paper" if args.paper else "user",
    )
    print(f"Conformal done ({', '.join(metrics)}) -> {out}")


if __name__ == "__main__":
    main()
