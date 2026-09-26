"""Build an auditable long-term validation supplement from the completed runs.

This script does not refit or alter the source data. It compares:
1) randomized Day-group nested CV (interpolation-oriented),
2) rolling-origin Day-group nested CV (future-date generalization), and
3) simple chronological baselines based only on earlier dates.

All prediction metrics are calculated after averaging the three technical
replicates within each specimen-day group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
PROJECT = PACKAGE_ROOT
WORKSPACE = PACKAGE_ROOT
STRICT_PROJECT = PACKAGE_ROOT / "03_outputs" / "_rerun_chronological"
RANDOM_PROJECT = PACKAGE_ROOT / "03_outputs" / "_rerun_randomized"
DATA_ROOT = PACKAGE_ROOT / "01_data" / "2 长期监测"
OUT = PACKAGE_ROOT / "03_outputs" / "_rerun_long_term_comparison"
EVALUATED_DAYS = (23.0, 28.0, 34.0, 43.0, 53.0, 67.0)


def latest_run(project: Path) -> Path:
    """Resolve the most recent completed run recorded by an analysis script."""
    return Path((project / "LATEST_RUN.txt").read_text(encoding="utf-8").strip())


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["True Load"].to_numpy(float)
    p = frame["Predicted Load"].to_numpy(float)
    abs_err = np.abs(y - p)
    return {
        "N independent specimen-day groups": len(frame),
        "R2": r2_score(y, p),
        "RMSE (kN)": mean_squared_error(y, p) ** 0.5,
        "MAE (kN)": mean_absolute_error(y, p),
        "Within +/-0.5 kN (%)": 100.0 * np.mean(abs_err <= 0.5),
        "Within +/-1.0 kN (%)": 100.0 * np.mean(abs_err <= 1.0),
        "Maximum absolute error (kN)": abs_err.max(),
    }


def grouped_predictions(run: Path, protocol: str) -> pd.DataFrame:
    src = pd.read_csv(run / "predictions" / "nested_oof_predictions.csv")
    src = src[src["Stage"].eq("long_term")].copy()
    grouped = (
        src.groupby(["Beam", "Model", "Group value"], as_index=False)
        .agg({"True Load": "mean", "Predicted Load": "mean"})
        .rename(columns={"Group value": "Day"})
    )
    grouped.insert(0, "Protocol", protocol)
    return grouped


def chronological_baselines() -> pd.DataFrame:
    rows: list[dict] = []
    for beam in range(1, 7):
        source = pd.read_excel(DATA_ROOT / f"longterm-Beam{beam}.xlsx", usecols=["Day", "Load"])
        daily = source.groupby("Day", as_index=False)["Load"].mean().sort_values("Day")
        for day in EVALUATED_DAYS:
            history = daily[daily["Day"] < day]
            actual = float(daily.loc[daily["Day"].eq(day), "Load"].iloc[0])
            previous = float(history.iloc[-1]["Load"])
            mean_prior = float(history["Load"].mean())
            slope, intercept = np.polyfit(history["Day"].to_numpy(float), history["Load"].to_numpy(float), 1)
            trend = float(intercept + slope * day)
            for model, prediction in (
                ("Persistence baseline", previous),
                ("Prior-mean baseline", mean_prior),
                ("Linear-time baseline", trend),
            ):
                rows.append({
                    "Protocol": "rolling-origin chronological baseline",
                    "Beam": beam,
                    "Model": model,
                    "Day": day,
                    "True Load": actual,
                    "Predicted Load": prediction,
                })
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (protocol, model), frame in predictions.groupby(["Protocol", "Model"], sort=False):
        rows.append({"Protocol": protocol, "Model": model, **metrics(frame)})
    return pd.DataFrame(rows).sort_values(["Protocol", "RMSE (kN)"])


def beam_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (protocol, model, beam), frame in predictions.groupby(["Protocol", "Model", "Beam"], sort=False):
        rows.append({"Protocol": protocol, "Model": model, "Beam": beam, **metrics(frame)})
    return pd.DataFrame(rows).sort_values(["Protocol", "Model", "Beam"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random = grouped_predictions(
        latest_run(RANDOM_PROJECT), "randomized Day-group nested CV"
    )
    rolling = grouped_predictions(
        latest_run(STRICT_PROJECT), "rolling-origin Day-group nested CV"
    )
    baselines = chronological_baselines()
    all_predictions = pd.concat([random, rolling, baselines], ignore_index=True)
    summary = summarize(all_predictions)
    by_beam = beam_summary(all_predictions)

    selected = summary[
        ((summary["Protocol"] == "randomized Day-group nested CV") & (summary["Model"] == "SVM"))
        | ((summary["Protocol"] == "rolling-origin Day-group nested CV") & (summary["Model"] == "RF"))
        | summary["Model"].str.contains("baseline")
    ].copy()

    all_predictions.to_csv(OUT / "group_mean_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "all_models_and_baselines.csv", index=False, encoding="utf-8-sig")
    by_beam.to_csv(OUT / "metrics_by_beam.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "selected_protocol_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Long-term grouped and chronological validation supplement",
        "",
        "## Design",
        "",
        "- Unit of independence: specimen-day; the three same-day acquisitions are averaged only after prediction.",
        "- Randomized Day-group nested CV: leakage-free interpolation-oriented sensitivity analysis.",
        "- Rolling-origin nested CV: Days 0, 1, 2, 4 and 6 form the initial observation window; Days 23, 28, 34, 43, 53 and 67 are tested sequentially using only earlier dates.",
        "- Baselines use only target values observed at earlier dates and are included to judge whether EMI models add predictive value beyond the slow temporal trend.",
        "",
        "## Selected pooled results",
        "",
        selected.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "The randomized grouped result estimates interpolation among observed monitoring dates without same-day replicate leakage. The rolling-origin result estimates the harder task of forward temporal generalization. A positive pooled R2 can still be influenced by between-beam differences; beam-specific results and absolute errors must therefore accompany it.",
        "",
        "The persistence baseline is a particularly important comparator because long-term prestress changes slowly after the initial relaxation period. If the EMI model does not outperform this baseline, the result supports preliminary state association but not a claim of superior future-date forecasting.",
    ]
    (OUT / "supplement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
