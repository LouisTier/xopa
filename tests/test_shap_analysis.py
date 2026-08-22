import numpy as np
import pandas as pd

from xopa.explain import shap_analysis


def test_explanation_diagnostics_cover_every_selected_feature():
    shap_values = pd.DataFrame(
        {
            "data_001": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            "data_002": [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
            "data_003": [0.2, 0.1, 0.3, -0.1, -0.3, -0.2],
        }
    )
    target = pd.Series(np.linspace(-1.0, 1.0, len(shap_values)))

    calculate = getattr(shap_analysis, "explanation_diagnostics", None)
    assert calculate is not None, "full-feature explanation diagnostics are missing"

    rank_agreement, output_monotonicity = calculate(
        shap_values,
        target,
        group_size=3,
        bin_width=0.5,
    )

    assert len(rank_agreement) == 2
    assert list(output_monotonicity.index) == list(shap_values.columns)
