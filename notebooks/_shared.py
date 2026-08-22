"""Shared setup for the analysis notebooks.

Everything the notebooks have in common is defined once here: repository
paths, recorded-campaign access, the journal figure style, the paper
notation, and the category/colour contracts. Each notebook imports what it
uses and calls ``init_notebook(<notebook name>)`` to obtain its figure
directory and ``savefig`` helper.
"""

import json
from pathlib import Path

import matplotlib as mpl
import pandas as pd


def _find_repo_root() -> Path:
    for cand in (Path.cwd(), *Path.cwd().parents):
        if (cand / "pyproject.toml").exists() and (cand / "xopa").is_dir():
            return cand
    raise FileNotFoundError(f"xopa repo root not found from {Path.cwd()}")


ROOT = _find_repo_root()
CAMPAIGN_DIR = (
    ROOT
    / "results/final-campaigns"
    / "dce-full-rffs-4backbones-xgb-analyses-v2-2026-08-11"
)
SUPERCON_CAMPAIGN_DIR = (
    ROOT
    / "results/final-campaigns"
    / "supercon-scale-full-rffs-4backbones-xgb-analyses-v2-2026-08-20"
)

BACKBONES = ["xgboost", "random_forest", "lightgbm", "mlp"]
BACKBONE_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}
OPERATORS = ["rfe", "mrmr", "lasso", "tree"]
OPERATOR_LABELS = {"rfe": "RFE", "mrmr": "mRMR", "lasso": "LASSO", "tree": "Tree imp."}
METHODS = ["scp", "cqr", "ens"]
METHOD_LABELS = {"scp": "m-SCP", "cqr": "m-CQR", "ens": "ENS"}

# Paper notation used in figure labels and table headers.
N = {
    "tau": r"$\tau$",
    "rho": r"$\rho$",
    "Q": r"$Q$",
    "psi": r"$\Psi$",
    "K": r"$K$",
    "d_star": r"$d_*$",
    "d_plus": r"$d_+$",
    "pi_j": r"$\pi_j$",
    "I_j": r"$I_j$",
    "M": r"$M$",
    "alpha": r"$\alpha$",
    "q_hat": r"$\hat{q}_{1-\alpha}$",
    "gamma": r"$\Gamma^{\alpha}(\mathbf{x}^t)$",
    "D_T": r"$\mathcal{D}^{T}$",
    "D_V": r"$\mathcal{D}^{V}$",
    "D_C": r"$\mathcal{D}^{C}$",
    "D_t": r"$\mathcal{D}^{t}$",
}

CATEGORY_COLORS = {
    "CTX": "#E69F00",
    "PV": "#56B4E9",
    "RMC": "#009E73",
    "WC": "#CC79A7",
    "PRS": "#0072B2",
}
CATEGORY_RANGES = {
    "CTX": (1, 4),
    "PV": (5, 103),
    "RMC": (104, 140),
    "WC": (141, 162),
    "PRS": (183, 609),
}
METHOD_COLORS = {"scp": "#0072B2", "cqr": "#D55E00", "ens": "#009E73"}
BACKBONE_COLORS = {
    "xgboost": "#0072B2",
    "random_forest": "#009E73",
    "lightgbm": "#D55E00",
    "mlp": "#CC79A7",
}
PROBE_COLOR = "#D55E00"

# Broad four-group post-hoc window selected from the exploratory SHAP profile.
# It covers the onset of the clearest signed change in data_474; it is a
# descriptive subset, not an estimated change point.
CHRONOLOGICAL_INTERVAL = (22500, 28499)

SINGLE_COL, ONE_HALF_COL, DOUBLE_COL = 3.54, 5.51, 7.48

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.1,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "figure.dpi": 110,
    }
)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def run_dir(backbone: str, kind: str, campaign: Path = CAMPAIGN_DIR) -> Path:
    """Resolve the recorded run directory of one campaign stage."""
    pattern = "*_comparison" if kind == "comparison" else f"*_{kind}_*"
    runs = sorted((campaign / backbone).glob(pattern))
    if not runs:
        raise FileNotFoundError(f"no {kind} run recorded for {backbone}")
    return runs[-1]


def jload(path: Path):
    """Load a JSON artifact."""
    return json.loads(Path(path).read_text())


def feature_category(name: str) -> str:
    """Map a data_NNN feature name to its process category."""
    suffix = int(name.split("_")[1])
    for category, (low, high) in CATEGORY_RANGES.items():
        if low <= suffix <= high:
            return category
    return "other"


def init_notebook(nb_name: str, campaign: Path = CAMPAIGN_DIR):
    """Create the notebook's figure directory and return (FIG_DIR, savefig)."""
    fig_dir = ROOT / "results/notebook-figures" / nb_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    def savefig(fig, stem: str) -> None:
        """Write PDF + PNG into the gitignored notebook-figures folder."""
        for suffix in (".pdf", ".png"):
            fig.savefig(fig_dir / f"{stem}{suffix}", bbox_inches="tight")
        print(f"saved: {fig_dir / stem}.pdf/.png")

    print(f"campaign: {campaign.name}")
    print(f"figures -> {fig_dir}")
    return fig_dir, savefig
