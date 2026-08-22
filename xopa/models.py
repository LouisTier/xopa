"""The four backbones: factories, fitting, evaluation, and importances.

``make_model`` builds an unfitted XGBoost, random-forest, LightGBM, or MLP
regressor from constructor parameters; ``fit_model`` trains it with early
stopping monitored on the validation split where the backbone supports it.
The calibration split never appears in this module (see :mod:`xopa.data` for
the split policy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgbm_early_stopping
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

if TYPE_CHECKING:
    import torch

ModelName = Literal["xgboost", "random_forest", "lightgbm", "mlp"]

#: Backbones whose fitted estimators expose native ``feature_importances_``.
TREE_BACKBONES = ("xgboost", "random_forest", "lightgbm")


def make_model(name: ModelName, params: dict[str, Any], seed: int):
    """Build an unfitted regressor from constructor parameters.

    Args:
        name: One of ``xgboost``, ``random_forest``, ``lightgbm``, ``mlp``.
        params: Constructor keyword arguments for the chosen estimator
            (early stopping is a fit-time concern, see :func:`fit_model`).
        seed: Value for the estimator's ``random_state``.

    Returns:
        An unfitted scikit-learn compatible regressor.
    """
    if name == "xgboost":
        return XGBRegressor(**params, random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(**params, random_state=seed)
    if name == "lightgbm":
        return LGBMRegressor(**params, random_state=seed, verbose=-1)
    if name == "mlp":
        return PaperMLPRegressor(**params, random_state=seed)
    raise ValueError(f"unknown model name: {name!r}")


def fit_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_rounds: int | None = 20,
):
    """Fit a model, early-stopping on the validation split where supported.

    - XGBoost and LightGBM early-stop on ``(x_val, y_val)`` when provided.
    - RandomForest fits plainly.
    - The MLP fits for its configured fixed epoch count.

    Args:
        model: Unfitted estimator from :func:`make_model`.
        x_train: Training features.
        y_train: Training target.
        x_val: Validation features (early-stopping monitor), or None.
        y_val: Validation target, or None.
        early_stopping_rounds: Patience; None disables early stopping.

    Returns:
        The fitted model (same object).
    """
    use_val = x_val is not None and y_val is not None and early_stopping_rounds
    model_package = type(model).__module__.partition(".")[0]
    if model_package == "xgboost":
        if use_val:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        else:
            model.set_params(early_stopping_rounds=None)
            model.fit(x_train, y_train)
    elif model_package == "lightgbm":
        if use_val:
            model.fit(
                x_train,
                y_train,
                eval_X=x_val,
                eval_y=y_val,
                eval_metric="rmse",
                callbacks=[lgbm_early_stopping(early_stopping_rounds, verbose=False)],
            )
        else:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    return model


def evaluate(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute MSE, RMSE, MAPE, and R2 of ``model`` on ``(x, y)``."""
    pred = model.predict(x)
    mse = float(mean_squared_error(y, pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def model_importances(
    model,
    x: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    kind: Literal["auto", "permutation"] = "auto",
    seed: int = 0,
) -> np.ndarray:
    """Return per-feature importances of a fitted model.

    ``auto`` uses the estimator's native ``feature_importances_`` when present
    and falls back to permutation importance otherwise (requires ``x, y``).

    Args:
        model: Fitted estimator.
        x: Data for permutation importance (required when used).
        y: Target for permutation importance (required when used).
        kind: ``auto`` or ``permutation``.
        seed: Seed for the permutation shuffles.

    Returns:
        Importance array aligned with the columns of the fitting data.
    """
    if kind == "auto" and hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if x is None or y is None:
        raise ValueError("permutation importance requires x and y")
    result = permutation_importance(
        model, x, y, n_repeats=5, random_state=seed, n_jobs=1
    )
    return np.asarray(result.importances_mean, dtype=float)


class PaperMLPRegressor(RegressorMixin, BaseEstimator):
    """PyTorch dense dropout regressor with a scikit-learn interface.

    ``n_layers`` hidden layers of ``n_units`` units, each followed by the
    selected activation and dropout, and a single linear output unit trained
    with mean squared error for a fixed number of epochs.
    ``feature_importances_`` exposes the normalized absolute input-to-output
    connection strengths.
    """

    def __init__(
        self,
        *,
        n_epochs: int = 107,
        n_layers: int = 3,
        n_units: int = 256,
        activation: str = "relu",
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        optimizer: str = "adam",
        loss: str = "mse",
        device: str = "cpu",
        n_threads: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_epochs = n_epochs
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.device = device
        self.n_threads = n_threads
        self.random_state = random_state

    def _validate_parameters(self) -> None:
        if self.n_epochs < 1 or self.n_layers < 1 or self.n_units < 1:
            raise ValueError("n_epochs, n_layers, and n_units must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}")
        if self.learning_rate <= 0.0 or self.batch_size < 1:
            raise ValueError("learning_rate and batch_size must be positive")
        if self.activation not in {"relu", "tanh"}:
            raise ValueError(f"unsupported MLP activation: {self.activation!r}")
        if self.optimizer != "adam" or self.loss != "mse":
            raise ValueError(
                "PaperMLPRegressor requires optimizer='adam' and loss='mse'"
            )
        if self.device != "cpu":
            raise ValueError("PaperMLPRegressor currently requires device='cpu'")
        if self.n_threads < 1:
            raise ValueError("n_threads must be positive")

    def _build_network(self, n_features: int) -> torch.nn.Sequential:
        import torch

        activation_type = torch.nn.ReLU if self.activation == "relu" else torch.nn.Tanh
        modules: list = []
        width = n_features
        for _ in range(self.n_layers):
            modules.extend(
                [
                    torch.nn.Linear(width, self.n_units),
                    activation_type(),
                    torch.nn.Dropout(self.dropout),
                ]
            )
            width = self.n_units
        modules.append(torch.nn.Linear(width, 1))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def _connection_importances(model: torch.nn.Sequential) -> np.ndarray:
        """Return normalized absolute input-to-output connection strength."""
        import torch

        layers = [module for module in model if isinstance(module, torch.nn.Linear)]
        path_strength = layers[-1].weight.detach().abs().reshape(-1)
        for layer in reversed(layers[:-1]):
            path_strength = path_strength @ layer.weight.detach().abs()
        values = path_strength.cpu().numpy().astype(float, copy=False)
        total = float(values.sum())
        if total == 0.0:
            return np.full(len(values), 1.0 / len(values), dtype=float)
        return values / total

    @staticmethod
    def _arrays(
        x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(x, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).reshape(-1)
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("X and y must have shapes (n, d) and (n,)")
        if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
            raise ValueError("X and y must contain the same positive number of rows")
        if features.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError("X and y must contain only finite values")
        return features, targets

    def fit(
        self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> PaperMLPRegressor:
        """Fit for exactly ``n_epochs`` using shuffled mini-batches."""
        import torch

        self._validate_parameters()
        features, targets = self._arrays(x, y)
        torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.random_state)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        model = self._build_network(features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        x_tensor = torch.as_tensor(features)
        y_tensor = torch.as_tensor(targets).reshape(-1, 1)

        self.loss_curve_: list[float] = []
        for epoch in range(self.n_epochs):
            model.train()
            order = torch.randperm(len(x_tensor), generator=generator)
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(model(x_tensor[batch]), y_tensor[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"paper MLP produced a non-finite loss at epoch {epoch + 1}"
                    )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch)
            self.loss_curve_.append(epoch_loss / len(order))

        model.eval()
        self.model_ = model
        self.n_features_in_ = features.shape[1]
        self.feature_importances_ = self._connection_importances(model)
        self.epochs_ = self.n_epochs
        if isinstance(x, pd.DataFrame):
            self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        return self

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict with dropout disabled and bounded inference batches."""
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("PaperMLPRegressor.fit must be called before predict")
        features = np.asarray(x, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (n, {self.n_features_in_}), got {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("X must contain only finite values")
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                batch = torch.as_tensor(features[start : start + self.batch_size])
                predictions.append(self.model_(batch).squeeze(-1).numpy())
        if not predictions:
            return np.empty(0, dtype=float)
        return np.concatenate(predictions).astype(float, copy=False)

    def __sklearn_is_fitted__(self) -> bool:
        """Report fitted state to scikit-learn utilities."""
        return hasattr(self, "model_")

    def _more_tags(self) -> dict[str, Any]:
        return {"poor_score": True}
