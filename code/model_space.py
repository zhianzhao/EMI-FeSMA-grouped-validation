"""Shared model definitions and predefined candidate configurations.

The analysis scripts import :func:`candidates` and :func:`make_estimator` from
this module.  Candidate configurations are intentionally finite and fixed in
advance; model selection is performed only inside each outer training split.
"""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


SEED = 42


def candidates(model: str) -> list[dict]:
    """Return the fixed candidate configurations evaluated for one model."""
    if model == "DT":
        return [
            {"max_depth": d, "min_samples_leaf": leaf,
             "min_samples_split": split, "criterion": criterion}
            for d, leaf, split, criterion in [
                (3, 1, 2, "squared_error"), (5, 1, 2, "squared_error"),
                (8, 1, 2, "squared_error"), (12, 1, 2, "squared_error"),
                (None, 1, 2, "squared_error"), (5, 2, 5, "squared_error"),
                (8, 2, 5, "squared_error"), (None, 2, 5, "squared_error"),
                (5, 3, 8, "absolute_error"), (8, 3, 8, "absolute_error"),
                (None, 3, 8, "absolute_error"),
                (None, 5, 8, "absolute_error"),
            ]
        ]
    if model == "SVM":
        return [
            {"scaler": scaler, "kernel": kernel, "C": c,
             "epsilon": eps, "gamma": gamma}
            for scaler, kernel, c, eps, gamma in [
                ("standard", "linear", 0.3, 0.03, "scale"),
                ("standard", "linear", 1, 0.03, "scale"),
                ("standard", "linear", 3, 0.01, "scale"),
                ("robust", "linear", 3, 0.03, "scale"),
                ("robust", "linear", 10, 0.01, "scale"),
                ("standard", "rbf", 1, 0.03, "scale"),
                ("standard", "rbf", 3, 0.03, "scale"),
                ("standard", "rbf", 10, 0.01, 0.03),
                ("standard", "rbf", 30, 0.01, 0.1),
                ("robust", "rbf", 3, 0.03, "scale"),
                ("robust", "rbf", 10, 0.01, 0.03),
                ("robust", "rbf", 30, 0.1, 0.1),
            ]
        ]
    if model == "RF":
        return [
            {"n_estimators": n, "max_depth": depth,
             "min_samples_leaf": leaf, "max_features": mf}
            for n, depth, leaf, mf in [
                (150, 5, 1, "sqrt"), (300, 5, 1, 0.7),
                (300, 10, 1, 1.0), (300, None, 1, 0.7),
                (150, 10, 2, "sqrt"), (300, 10, 2, 0.7),
                (300, None, 2, 1.0), (150, 5, 4, "sqrt"),
                (300, 10, 4, 0.7), (300, None, 4, 1.0),
            ]
        ]
    if model == "MLP":
        return [
            {"scaler": scaler, "hidden_layer_sizes": hidden,
             "activation": activation, "alpha": alpha}
            for scaler, hidden, activation, alpha in [
                ("standard", (5,), "tanh", 0.01),
                ("standard", (10,), "tanh", 0.1),
                ("standard", (20,), "tanh", 1.0),
                ("standard", (10, 5), "tanh", 0.1),
                ("robust", (5,), "tanh", 0.01),
                ("robust", (10,), "tanh", 0.1),
                ("robust", (20,), "tanh", 1.0),
                ("standard", (10,), "relu", 0.1),
                ("standard", (20,), "relu", 1.0),
                ("robust", (10, 5), "relu", 1.0),
            ]
        ]
    if model == "XGBoost":
        return [
            {"n_estimators": n, "max_depth": depth, "learning_rate": lr,
             "min_child_weight": child, "subsample": subsample,
             "colsample_bytree": colsample, "reg_lambda": reg_lambda}
            for n, depth, lr, child, subsample, colsample, reg_lambda in [
                (100, 2, 0.03, 1, 1.0, 1.0, 1),
                (200, 2, 0.05, 1, 0.8, 0.8, 5),
                (300, 2, 0.05, 3, 0.8, 1.0, 10),
                (200, 3, 0.03, 3, 1.0, 0.8, 5),
                (300, 3, 0.05, 1, 0.8, 0.8, 5),
                (200, 3, 0.1, 3, 0.8, 1.0, 10),
                (300, 3, 0.03, 5, 1.0, 1.0, 10),
                (200, 4, 0.03, 3, 0.8, 0.8, 5),
                (300, 4, 0.05, 5, 1.0, 0.8, 10),
                (100, 4, 0.1, 1, 0.8, 1.0, 5),
            ]
        ]
    raise KeyError(model)


def make_estimator(model: str, cfg: dict):
    """Construct an estimator without fitting it."""
    if model == "DT":
        return DecisionTreeRegressor(random_state=SEED, **cfg)
    if model == "RF":
        return RandomForestRegressor(
            random_state=SEED, n_jobs=1, **cfg
        )
    if model == "XGBoost":
        return XGBRegressor(
            random_state=SEED,
            n_jobs=1,
            objective="reg:squarederror",
            verbosity=0,
            reg_alpha=0.0,
            **cfg,
        )

    scaler = (
        StandardScaler() if cfg["scaler"] == "standard" else RobustScaler()
    )
    if model == "SVM":
        regressor = SVR(
            kernel=cfg["kernel"],
            C=cfg["C"],
            epsilon=cfg["epsilon"],
            gamma=cfg["gamma"],
        )
    elif model == "MLP":
        regressor = MLPRegressor(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation=cfg["activation"],
            solver="lbfgs",
            alpha=cfg["alpha"],
            max_iter=5000,
            random_state=SEED,
        )
    else:
        raise KeyError(model)

    pipeline = Pipeline([("scaler", scaler), ("regressor", regressor)])
    return TransformedTargetRegressor(
        regressor=pipeline, transformer=StandardScaler()
    )
