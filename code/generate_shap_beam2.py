"""Generate manuscript SHAP figures for the final Beam 2 short-term XGBoost model.

The model is fitted to the complete Beam 2 short-term dataset using the
representative hyperparameters selected by grouped inner cross-validation.
Exact TreeSHAP contributions are obtained directly from the XGBoost booster.
This analysis explains the fitted beam-specific model; it is not a performance
estimate and it is not pooled across beams.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
DATA_PATH = PACKAGE_ROOT / "01_data" / "1 短期加载" / "Database-Beam2.xlsx"
OUTPUT_ROOT = PACKAGE_ROOT / "03_outputs" / "06_shap_short_term_beam2"
FIGURE_ROOT = (
    PACKAGE_ROOT
    / "04_review_material"
    / "02_manuscript_figures"
    / "06_SHAP_short_term_Beam2"
)

FEATURES = ["RMSD", "Peak", "Frequency"] + [f"Area{i}" for i in range(1, 13)]
TARGET = "Load"
SEED = 42
MODEL_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
}


def display_name(feature: str) -> str:
    if feature.startswith("Area"):
        return feature.replace("Area", "Area ")
    return feature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_data() -> pd.DataFrame:
    data = pd.read_excel(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data[required].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        bad_rows = data.index[data.isna().any(axis=1)].tolist()
        raise ValueError(f"Missing or nonnumeric values in rows: {bad_rows}")
    if not np.isfinite(data.to_numpy(float)).all():
        raise ValueError("Dataset contains non-finite values")
    return data.reset_index(drop=True)


def fit_and_explain(data: pd.DataFrame) -> tuple[XGBRegressor, np.ndarray, np.ndarray]:
    x_values = data[FEATURES].to_numpy(float)
    y_values = data[TARGET].to_numpy(float)
    model = XGBRegressor(
        random_state=SEED,
        n_jobs=1,
        objective="reg:squarederror",
        verbosity=0,
        reg_alpha=0.0,
        **MODEL_CONFIG,
    )
    model.fit(x_values, y_values)
    contributions = model.get_booster().predict(
        xgb.DMatrix(x_values, feature_names=FEATURES),
        pred_contribs=True,
        approx_contribs=False,
    )
    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]
    predictions = model.predict(x_values)
    residual = float(np.max(np.abs(contributions.sum(axis=1) - predictions)))
    if residual > 1e-4:
        raise RuntimeError(f"TreeSHAP additivity check failed: {residual}")
    return model, shap_values, base_values


def symmetric_offsets(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    if span == 0:
        return np.zeros_like(values)
    bins = np.floor((values - values.min()) / span * 24).astype(int)
    rng = np.random.default_rng(seed)
    offsets = np.zeros_like(values)
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        rng.shuffle(indices)
        sequence = [0.0]
        for level in range(1, len(indices)):
            distance = (level + 1) // 2
            sign = 1.0 if level % 2 else -1.0
            sequence.append(sign * distance * 0.060)
        offsets[indices] = sequence
    return np.clip(offsets, -0.30, 0.30)


def normalized_colors(values: np.ndarray) -> np.ndarray:
    low, high = np.quantile(values, [0.05, 0.95])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full_like(values, 0.5, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def plot_summary(
    shap_values: np.ndarray,
    raw_features: np.ndarray,
    order: np.ndarray,
    output_stem: Path,
) -> None:
    color_map = LinearSegmentedColormap.from_list(
        "shap_blue_magenta", ["#1685d1", "#7f62c9", "#ff1764"]
    )
    figure, axis = plt.subplots(figsize=(5.35, 5.75))
    figure.subplots_adjust(left=0.24, right=0.86, bottom=0.13, top=0.97)

    for row, feature_index in enumerate(order):
        x_values = shap_values[:, feature_index]
        y_values = row + symmetric_offsets(x_values, SEED + int(feature_index))
        colors = normalized_colors(raw_features[:, feature_index])
        axis.scatter(
            x_values,
            y_values,
            c=colors,
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
            s=28,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    axis.axvline(0.0, color="#666666", linewidth=0.9, zorder=1)
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([display_name(FEATURES[index]) for index in order])
    axis.invert_yaxis()
    axis.set_xlabel("SHAP value (impact on model output, kN)")
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.50, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    color_scalar = mpl.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=color_map)
    color_scalar.set_array([])
    colorbar = figure.colorbar(color_scalar, ax=axis, pad=0.03, fraction=0.05)
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["Low", "High"])
    colorbar.set_label("Feature value", rotation=270, labelpad=16)

    for suffix in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def plot_importance(importance: np.ndarray, order: np.ndarray, output_stem: Path) -> None:
    ordered_values = importance[order]
    names = [display_name(FEATURES[index]) for index in order]
    figure, axis = plt.subplots(figsize=(5.20, 5.75))
    figure.subplots_adjust(left=0.24, right=0.97, bottom=0.13, top=0.97)
    bars = axis.barh(
        np.arange(len(order)),
        ordered_values,
        color="#168bea",
        edgecolor="#0f6fb7",
        linewidth=0.5,
        height=0.68,
    )
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels(names)
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Mean absolute SHAP value (kN)")
    axis.grid(axis="x", color="#d0d0d0", linewidth=0.50, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.16)

    maximum = max(float(ordered_values.max()), 1e-9)
    for bar, value in zip(bars, ordered_values):
        axis.text(
            bar.get_width() + maximum * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=9.5,
        )

    for suffix in ("png", "pdf"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=600 if suffix == "png" else None,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def plot_preview(summary_png: Path, importance_png: Path, output_path: Path) -> None:
    summary = plt.imread(summary_png)
    importance = plt.imread(importance_png)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.6))
    for axis, bitmap, label in zip(
        axes,
        (summary, importance),
        ("(a) SHAP summary", "(b) Mean absolute SHAP values"),
    ):
        axis.imshow(bitmap)
        axis.axis("off")
        axis.set_title(label, fontsize=13, pad=3)
    figure.tight_layout(pad=0.25)
    figure.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def export_source_data(
    data: pd.DataFrame,
    model: XGBRegressor,
    shap_values: np.ndarray,
    base_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = model.predict(data[FEATURES].to_numpy(float))
    shap_frame = data.copy()
    for feature_index, feature in enumerate(FEATURES):
        shap_frame[f"SHAP_{feature}"] = shap_values[:, feature_index]
    shap_frame["TreeSHAP_base_value"] = base_values
    shap_frame["Fitted_prediction_kN"] = predictions
    shap_frame["TreeSHAP_sum_kN"] = base_values + shap_values.sum(axis=1)
    shap_frame.to_csv(
        OUTPUT_ROOT / "shap_values_Beam2_XGBoost.csv", index=False, encoding="utf-8-sig"
    )

    importance = np.mean(np.abs(shap_values), axis=0)
    importance_frame = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Mean absolute SHAP value (kN)": importance,
            }
        )
        .sort_values("Mean absolute SHAP value (kN)", ascending=False)
        .reset_index(drop=True)
    )
    importance_frame["Rank"] = np.arange(1, len(importance_frame) + 1)
    importance_frame.to_csv(
        OUTPUT_ROOT / "shap_importance_Beam2_XGBoost.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return shap_frame, importance_frame


def main() -> None:
    configure_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    data = read_data()
    model, shap_values, base_values = fit_and_explain(data)
    shap_frame, importance_frame = export_source_data(
        data, model, shap_values, base_values
    )
    importance = importance_frame.set_index("Feature").loc[FEATURES][
        "Mean absolute SHAP value (kN)"
    ].to_numpy(float)
    order = np.argsort(importance)[::-1]

    summary_stem = FIGURE_ROOT / "Fig13_SHAP_summary_Beam2_XGBoost"
    importance_stem = FIGURE_ROOT / "Fig14_SHAP_importance_Beam2_XGBoost"
    plot_summary(
        shap_values,
        data[FEATURES].to_numpy(float),
        order,
        summary_stem,
    )
    plot_importance(importance, order, importance_stem)
    plot_preview(
        summary_stem.with_suffix(".png"),
        importance_stem.with_suffix(".png"),
        FIGURE_ROOT / "Fig13_14_SHAP_Beam2_preview.png",
    )

    predictions = model.predict(data[FEATURES].to_numpy(float))
    additivity_error = float(
        np.max(np.abs(shap_frame["TreeSHAP_sum_kN"].to_numpy() - predictions))
    )
    manifest = {
        "analysis": "Beam 2 short-term final-model TreeSHAP",
        "interpretation_scope": "One beam-specific fitted model; not a performance estimate",
        "data_path": str(DATA_PATH.relative_to(PACKAGE_ROOT)),
        "data_sha256": sha256(DATA_PATH),
        "rows": len(data),
        "distinct_load_groups": int(data[TARGET].nunique()),
        "features": FEATURES,
        "target": TARGET,
        "model": "XGBoost",
        "model_config": MODEL_CONFIG,
        "reg_alpha": 0.0,
        "random_seed": SEED,
        "augmentation": "none",
        "tree_shap": "exact pred_contribs from fitted XGBoost booster",
        "max_additivity_error": additivity_error,
        "feature_ranking": importance_frame.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Rows: {len(data)}")
    print(f"Distinct Load groups: {data[TARGET].nunique()}")
    print(f"Maximum TreeSHAP additivity error: {additivity_error:.3e}")
    print(importance_frame.to_string(index=False))
    print(f"Figures: {FIGURE_ROOT}")
    print(f"Source data: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
