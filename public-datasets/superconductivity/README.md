# UCI Superconductivity dataset (public demonstration)

Public, redistributable dataset used to demonstrate the xopa pipeline (RFFS feature selection, m-SCP / m-CQR conformal prediction, SHAP explainability) on open data.

## Contents

| File | Description |
| --- | --- |
| `train.csv` | Original UCI file: 21,263 rows, 81 numeric material-property features, continuous target `critical_temp` (Kelvin) |
| `xopa/superconductivity_xopa.csv` | xopa-schema conversion produced by `scripts/prepare_uci_superconductivity.py`: features renamed `data_001`..`data_081`, target retained in Kelvin and named `data_164` |
| `xopa/superconductivity_mapping.csv` | Original-name to `data_NNN` mapping (the source of truth for interpreting results; the CTX/PV category labels carry no meaning for this dataset) |
| `xopa/superconductivity_target_metadata.json` | Target name, stored unit, and storage-transform metadata |

The public campaign uses `configs/superconductivity.yaml`.  
It first draws the fixed four-way split, then estimates the target mean and population standard deviation on the training rows only.  
That transform is applied unchanged to the validation, calibration, and test targets, and its fitted values are saved in each run's `preprocessing.npz` artifact.

## Source, license, attribution

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/464 (doi:10.24432/C53P47), licensed **CC BY 4.0**.
- Please cite: Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. Computational Materials Science, 154, 346-354.

The proprietary industrial dataset of the accompanying paper is not distributed anywhere in this repository.  
This folder contains only public, redistributable data.
