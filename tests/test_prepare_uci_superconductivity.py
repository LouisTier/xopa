import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_uci_superconductivity.py"
_SPEC = importlib.util.spec_from_file_location("prepare_uci_superconductivity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert = _MODULE.convert


def test_conversion_keeps_the_public_target_in_kelvin():
    source = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "critical_temp": [4.5, 18.0, 92.0],
        }
    )

    converted, _, metadata = convert(source)

    pd.testing.assert_series_equal(
        converted["data_164"],
        source["critical_temp"].rename("data_164"),
    )
    assert metadata["stored_unit"] == "kelvin"
    assert metadata["storage_transform"] == "identity"
    assert "mean" not in metadata
    assert "std" not in metadata
