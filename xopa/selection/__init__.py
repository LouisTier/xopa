"""Feature selection: RFFS (FFS + SFS), ranking operators, probes, Boruta.

- :mod:`xopa.selection.ffs`: Frequency Feature Selection: Q subsample draws,
  one ranking each, survivors by selection frequency over a d-grid.
- :mod:`xopa.selection.sfs`: SHAP Feature Selection: SHAP-ranked top-k
  validation sweep yielding the optimal and parsimonious subset sizes.
- :mod:`xopa.selection.rffs`: the composition of both plus the consolidated
  report (winner cell, test metrics, category table, overlaps).
- :mod:`xopa.selection.rankers`: the interchangeable ranking operators (RFE,
  mRMR, LASSO, tree importance), all honoring a nested-prefix contract.
- :mod:`xopa.selection.probes`: synthetic noise probes for the data-determined
  threshold and noise floor.
- :mod:`xopa.selection.boruta`: canonical Boruta as a standalone baseline.
"""

from xopa.selection.boruta import BorutaResult, run_boruta
from xopa.selection.ffs import FFSResult, run_ffs
from xopa.selection.probes import ProbeConfig, add_probes, is_probe
from xopa.selection.rankers import RANKERS
from xopa.selection.rffs import RFFSReport, jaccard_overlap, run_rffs
from xopa.selection.sfs import SFSResult, run_sfs

__all__ = [
    "RANKERS",
    "BorutaResult",
    "FFSResult",
    "ProbeConfig",
    "RFFSReport",
    "SFSResult",
    "add_probes",
    "is_probe",
    "jaccard_overlap",
    "run_boruta",
    "run_ffs",
    "run_rffs",
    "run_sfs",
]
