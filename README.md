# xopa

Explainable Offline Process Analytics for tabular regression: **RFFS** feature selection, **m-SCP / m-CQR** multi-split conformal prediction, and SHAP explainability with noise-floor diagnostics.

Companion code for:

> *Explainable Offline Process Analytics for Within-Grade Quality Variability in Industrial Rubber Mixing*, Digital Chemical Engineering, 2026.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) on Apple-silicon macOS or Linux.

```bash
git clone https://github.com/LouisTier/xopa.git
cd xopa
make sync
make demo
```

- `make demo` runs a quick end-to-end check of all five stages on the bundled [UCI Superconductivity dataset](https://archive.ics.uci.edu/dataset/464) (21,263 records, 81 features, CC BY 4.0; see `public-datasets/superconductivity/README.md`), so nothing needs to be downloaded or configured; it takes about seven minutes on a laptop.
The demo uses a reduced selection grid, 500 conformal shuffles, and one XGBoost selection campaign so that the complete artifact path can be checked quickly.
- The paper's recorded public campaign uses the full settings in `configs/superconductivity.yaml`. Its executed analysis, including the reported tables and visualizations, is retained in `notebooks/08_superconductivity.ipynb`.
Each demo stage writes a timestamped directory under `results/demo/` with its artifacts and a `manifest.json` recording the resolved configuration, source and environment metadata, package versions, and any seed metadata supplied by that stage.

| Stage | Script | Question it answers |
| --- | --- | --- |
| Feature selection | `run_rffs.py` | Which features survive frequency selection (FFS), and where does the SHAP refinement sweep (SFS) place the optimal and parsimonious subset sizes? |
| Baselines | `run_baselines.py` | What do Boruta and one-shot rankers select, for comparison? |
| Conformal | `run_conformal.py` | How wide must prediction intervals be for 90% coverage (m-SCP, m-CQR, and an ensemble baseline)? |
| Explainability | `run_explain.py` | Which features drive predictions, and is that stable across the dataset? |
| Model comparison | `run_model_comparison.py` | How do the four backbones compare on the selected feature sets? |

## Using your own data

Every script takes an explicit `--data` path to a CSV or Parquet file with: feature columns named `data_NNN`, a continuous target column (default `data_164`, override with the `data.target` config key), optional sparse `target_*` columns (dropped), and an optional serialized index column (dropped).  
To adapt an arbitrary table, write a small renaming script like `scripts/prepare_uci_superconductivity.py` and keep the name mapping next to the converted file.  
Set `data standardize_target: true` when the source target must be standardized.  
The split is drawn first, the transform is fitted on the training target only, and the same transform is applied to validation, calibration, and test.

Run stages through the Makefile (extra flags go in `ARGS`):

```bash
make rffs DATA=/abs/path/dataset.csv BACKBONE=xgboost
make conformal DATA=/abs/path/dataset.csv ARGS="--features-file results/<run>/rffs_summary.json"
```

or call the scripts directly; `--help` on any script lists the overrides, and `configs/default.yaml` documents every setting in one place.

The package is also usable directly, with explicit data arguments at every stage, so any intermediate result can be replaced by your own:

```python
from xopa.data import load_dataset, make_splits
from xopa.selection import run_rffs
from xopa.conformal import fit_mscp, predict_mscp

x, y = load_dataset("dataset.csv")
splits = make_splits(x, y, seed=42)
report = run_rffs(
    splits,
    operators=["rfe", "tree"],
    grid=[70, 50, 30],
    backbone="xgboost",
    model_params={"n_estimators": 172, "max_depth": 9, "learning_rate": 0.076},
)
```

## Package tour

| Location | What lives there |
| --- | --- |
| `xopa/data.py` | Dataset loading, the fixed train/val/calib/test split (70/7.5/7.5/15), feature categories, dataset-identity validation |
| `xopa/models.py` | The four backbones (XGBoost, random forest, LightGBM, PyTorch MLP) behind one factory, fitting, and evaluation interface |
| `xopa/selection/` | FFS (Algorithm 1) with interchangeable ranking operators (RFE, mRMR, LASSO, tree importance), synthetic noise probes, SFS refinement, standalone Boruta, RFFS orchestration |
| `xopa/conformal/` | m-SCP and m-CQR (Algorithms 2 and 3) with `m`, `alpha`, and the split fraction as parameters; a bootstrap ensemble baseline; PICP / MPIW / WCC metrics |
| `xopa/explain/` | Full-dataset SHAP attributions, record-index drift, rank-stability and monotonicity statistics, per-category importance |
| `xopa/utils.py` | Configuration loading, derived seeds, run directories, and run manifests |
| `scripts/` | Five thin CLIs (one per stage) plus the public-dataset adapter |
| `configs/default.yaml` | The paper settings, with target values left unchanged |
| `configs/superconductivity.yaml` | Public-data settings, including train-fitted target standardization |
| `public-datasets/` | Bundled open datasets with attribution and license notes |
| `notebooks/` | Executed records of the paper analyses; saved outputs remain directly viewable, while rerunning campaign-dependent cells requires the corresponding retained artifacts. Common setup lives in `notebooks/_shared.py` |

The split policy is strict throughout: train fits, validation selects features and parameters (and monitors early stopping), no model is ever fitted, selected, or tuned on the calibration split (only `xopa.conformal` consumes it; post-fit reporting such as full-dataset SHAP covers all splits), and test is touched once.  
Two implementation details worth knowing:

- **Nested-prefix rankers.** Every ranking operator returns an ordering whose first K entries equal its direct selection at size K, so one ranking per (subsample, operator) serves the whole d-grid: changing the grid never requires re-running a simulation.
- **Noise probes.** Synthetic columns (Gaussian, and permuted copies of real features) ride through selection; their selection frequency and SHAP values provide a data-determined threshold (`tau: auto`) and a noise-floor line on the refinement curve.

## Recorded artifacts

A completed run directory contains the inputs needed to rebuild its figures, tables, and numerical summaries without refitting that stage.
Alongside `manifest.json`, `split_assignments.csv`, `preprocessing.npz`, and `dataset_identity.json`, each stage records:

| Stage | Key artifacts |
| --- | --- |
| rffs | Per-operator grids, selection counts and probe counts, full rankings, `selection_frequency.csv`, SFS curve and importance, survivor sets, `rffs_summary.json`, per-sample test predictions |
| baselines | Boruta decisions and importance history, one-shot selections, validation metrics, Jaccard overlaps |
| conformal | Per-sample test intervals, per-split thresholds and coverages, m-checkpoints, calibration resampling arrays, `uq_metrics.json` with regime-conditional coverage |
| explain | Full-dataset SHAP bundle (values, feature values, predictions, split labels), drift tables, stability and monotonicity statistics, category summary |
| comparison | Per-sample predictions and metrics for every backbone and feature set |

This table documents the artifact contract produced by a run.  
The industrial final-campaign directories contain record-level derived values and are not part of the public source repository. They remain under Michelin confidentiality.  
The bundled UCI data and `make demo` generate the same staged artifact contract from public data, with the lighter settings described in the quick start. This is the clean-checkout execution path for external users. The executed notebooks preserve the full reported analyses and visualizations, but they are records of retained campaigns.

## Reproducing the paper (authors)

The industrial dataset of the paper is proprietary and **not distributed**; no proprietary data file is stored in this repository. The `--paper` flag (and the `paper-*` Make targets) validates the loaded dataset against the paper's recorded identity (source hash, shapes, dropped columns) before anything runs, and labels the run's manifest accordingly.

`configs/default.yaml` is the complete hyperparameter contract of the paper: 

- XGBoost at 172 estimators and depth 9,  
- the split ratios and seed,  
- Q=10, rho=0.7, tau=0.8, the matched grid `[170, 150, 130, 110, 90, 70, 50, 30]`,  
- alpha=0.1, and M=2,200  

are all recorded in the same file.

```bash
make paper-rffs DATA=/abs/path/dataset.csv          # RFFS for all four backbones
make paper-no-prs DATA=/abs/path/dataset.csv        # XGBoost pipeline without PRS features
make paper-analysis DATA=/abs/path/dataset.csv RFFS_SUMMARY=results/<run>/rffs_summary.json
```

## Development

```bash
make lint   # ruff format --check + ruff check
make test   # pytest (small, synthetic-data test suite)
```

## License and citation

Apache-2.0 (see `LICENSE`). If you use this code, please cite the paper, see `CITATION.cff`.
