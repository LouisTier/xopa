"""Convert the UCI Superconductivity dataset to the xopa schema.

Public-dataset demonstration, licensed CC BY 4.0:
``train.csv`` (21,263 rows; 81 numeric material-property features and the
continuous target ``critical_temp`` in Kelvin) becomes an xopa-compatible
table. The target remains in Kelvin and is named ``data_164``; target
standardization, when requested by the analysis configuration, is fitted on
the training split only. Features are renamed ``data_001``..``data_081`` in
original column order. The synthetic category labels carry no meaning here;
the mapping file is the source of truth for interpretation.

Dataset: https://archive.ics.uci.edu/dataset/464 (Hamidieh, 2018).

Example:
    uv run python scripts/prepare_uci_superconductivity.py \
        --train /path/to/train.csv --out /path/to/superconductivity-xopa
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TARGET_NAME = "data_164"
ORIGINAL_TARGET = "critical_temp"


def convert(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return the Kelvin-valued table, feature mapping, and target metadata."""
    if ORIGINAL_TARGET not in train.columns:
        raise ValueError(f"input is missing the target column {ORIGINAL_TARGET!r}")
    features = train.drop(columns=[ORIGINAL_TARGET])

    mapping = pd.DataFrame(
        {
            "original": features.columns,
            "xopa_name": [f"data_{i:03d}" for i in range(1, len(features.columns) + 1)],
            "encoding": "identity",
        }
    )
    converted = features.astype(float)
    converted.columns = list(mapping["xopa_name"])

    target_metadata = {
        "original_target": ORIGINAL_TARGET,
        "xopa_target": TARGET_NAME,
        "source_unit": "kelvin",
        "stored_unit": "kelvin",
        "storage_transform": "identity",
        "analysis_transform": "configured by the pipeline and fitted on the training split only",
    }
    converted[TARGET_NAME] = train[ORIGINAL_TARGET].astype(float)
    return converted, mapping, target_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="UCI train.csv path")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    converted, mapping, target_metadata = convert(pd.read_csv(args.train))

    converted.to_csv(out / "superconductivity_xopa.csv", index=False)
    mapping.to_csv(out / "superconductivity_mapping.csv", index=False)
    (out / "superconductivity_target_metadata.json").write_text(
        json.dumps(target_metadata, indent=2) + "\n"
    )
    n_features = converted.shape[1] - 1
    print(
        f"converted: {converted.shape[0]} rows, {n_features} features + "
        f"{TARGET_NAME} ({ORIGINAL_TARGET}, Kelvin) -> "
        f"{out / 'superconductivity_xopa.csv'}"
    )


if __name__ == "__main__":
    main()
