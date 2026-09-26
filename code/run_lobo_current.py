"""Nested leave-one-beam-out validation for the current 0924 datasets.

Outer unit: one complete beam.
Inner model selection: leave one of the remaining beams out.
Analysis unit: one Beam x physical-state group after averaging technical replicates.

The analysis uses the same 15 impedance features as the specimen-specific
models. The different historical sweep ranges are retained as a stated
limitation of cross-specimen interpretation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_ROOT = ROOT / "01_data"
OUTPUT_ROOT = ROOT / "03_outputs" / "04_lobo_validation"

SEED = 42
BOOTSTRAP_REPS = 3000
BEAMS = tuple(range(1, 7))
MODELS = ("DT", "SVM", "RF", "MLP", "XGBoost")
FULL_FEATURES = ["RMSD", "Peak", "Frequency"] + [f"Area{i}" for i in range(1, 13)]
FEATURE_SETS = {"Full15": FULL_FEATURES}
STAGES = {
    "short_term": {
        "folder": "1 短期加载",
        "file": "Database-Beam{beam}.xlsx",
        "group": "Load",
    },
    "long_term": {
        "folder": "2 长期监测",
        "file": "longterm-Beam{beam}.xlsx",
        "group": "Day",
    },
    "failure": {
        "folder": "3 加载破坏",
        "file": "failure-Beam{beam}.xlsx",
        "group": "Applied load",
    },
}

helper_path = HERE / "model_space.py"
spec = importlib.util.spec_from_file_location("lobo_current_model_space", helper_path)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(stage: str, beam: int) -> Path:
    cfg = STAGES[stage]
    return DATA_ROOT / cfg["folder"] / cfg["file"].format(beam=beam)


def load_grouped_stage(stage: str) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    inventory = []
    group_column = STAGES[stage]["group"]
    for beam in BEAMS:
        path = source_path(stage, beam)
        raw = pd.read_excel(path)
        required = FULL_FEATURES + ["Load"]
        if group_column != "Load":
            required.append(group_column)
        missing = [column for column in required if column not in raw.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        numeric = raw[required].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            rows = numeric.index[numeric.isna().any(axis=1)].tolist()
            raise ValueError(f"{path}: non-numeric/missing values at rows {rows}")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{path}: non-finite values")

        conflicts = numeric.groupby(group_column)["Load"].nunique(dropna=False)
        if (conflicts > 1).any():
            raise ValueError(f"{path}: target conflicts inside {group_column} groups")

        aggregated = (
            numeric.groupby(group_column, as_index=False)
            .agg(
                **{feature: (feature, "mean") for feature in FULL_FEATURES},
                Load=("Load", "mean"),
                technical_replicates=("Load", "size"),
            )
            .rename(columns={group_column: "Group value"})
        )
        if group_column == "Load":
            # pandas cannot retain the grouping key and an aggregation output
            # with the same name; for short-term data the group value is the target.
            aggregated["Load"] = aggregated["Group value"]
        aggregated["Stage"] = stage
        aggregated["Beam"] = beam
        aggregated["Group column"] = group_column
        frames.append(aggregated)
        inventory.append(
            {
                "Stage": stage,
                "Beam": beam,
                "Source file": str(path.relative_to(ROOT)),
                "SHA256": sha256(path),
                "Raw rows": len(numeric),
                "Independent state groups": len(aggregated),
                "Group column": group_column,
                "Target minimum (kN)": float(numeric["Load"].min()),
                "Target maximum (kN)": float(numeric["Load"].max()),
            }
        )
    return pd.concat(frames, ignore_index=True), inventory


def fit_predict(model: str, cfg: dict, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    estimator = helper.make_estimator(model, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(train[features].to_numpy(float), train["Load"].to_numpy(float))
    prediction = np.asarray(estimator.predict(test[features].to_numpy(float)), float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    return estimator, prediction


def metric_values(y, prediction) -> dict[str, float]:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return {
        "R2": float(r2_score(y, prediction)) if len(y) > 1 and np.unique(y).size > 1 else np.nan,
        "RMSE (kN)": float(math.sqrt(mean_squared_error(y, prediction))),
        "MAE (kN)": float(mean_absolute_error(y, prediction)),
    }


def tune_on_training_beams(
    model: str,
    training: pd.DataFrame,
    features: list[str],
) -> tuple[dict, float, list[dict]]:
    training_beams = sorted(training["Beam"].unique())
    candidate_rows = []
    best_cfg = None
    best_rmse = np.inf
    for candidate_index, cfg in enumerate(helper.candidates(model), start=1):
        fold_rmse = []
        failed = False
        for validation_beam in training_beams:
            inner_train = training[training["Beam"] != validation_beam]
            inner_test = training[training["Beam"] == validation_beam]
            try:
                _, prediction = fit_predict(model, cfg, inner_train, inner_test, features)
                fold_rmse.append(metric_values(inner_test["Load"], prediction)["RMSE (kN)"])
            except Exception:
                failed = True
                break
        mean_rmse = float(np.mean(fold_rmse)) if fold_rmse and not failed else np.inf
        candidate_rows.append(
            {
                "Candidate": candidate_index,
                "Mean inner held-out-beam RMSE (kN)": mean_rmse,
                "Valid inner folds": len(fold_rmse),
                "Configuration": json.dumps(cfg, sort_keys=True),
            }
        )
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_cfg = cfg.copy()
    if best_cfg is None:
        raise RuntimeError(f"No valid candidate for {model}")
    return best_cfg, float(best_rmse), candidate_rows


def cluster_bootstrap_ci(frame: pd.DataFrame, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    beams = np.array(sorted(frame["Held-out Beam"].unique()))
    samples = {"R2": [], "RMSE (kN)": [], "MAE (kN)": []}
    for _ in range(BOOTSTRAP_REPS):
        selected = rng.choice(beams, size=len(beams), replace=True)
        parts = [frame[frame["Held-out Beam"] == beam] for beam in selected]
        sampled = pd.concat(parts, ignore_index=True)
        score = metric_values(sampled["True Load"], sampled["Predicted Load"])
        for key, value in score.items():
            if np.isfinite(value):
                samples[key].append(value)
    result = {}
    for key, values in samples.items():
        result[f"{key} CI lower"] = float(np.percentile(values, 2.5)) if values else np.nan
        result[f"{key} CI upper"] = float(np.percentile(values, 97.5)) if values else np.nan
    return result


def read_within_specimen_metrics() -> pd.DataFrame:
    primary_path = ROOT / "03_outputs" / "01_randomized_group_primary" / "tables" / "pooled_group_metrics.csv"
    strict_path = (
        ROOT
        / "03_outputs"
        / "02_grouped_failure_and_chronological_sensitivity"
        / "tables"
        / "pooled_group_metrics.csv"
    )
    primary = pd.read_csv(primary_path)
    strict = pd.read_csv(strict_path)
    selected = pd.concat(
        [
            primary[primary["Stage"].isin(["short_term", "long_term"])],
            strict[strict["Stage"] == "failure"],
        ],
        ignore_index=True,
    )
    return selected[
        [
            "Stage",
            "Model",
            "Independent groups",
            "Pooled group-mean R2",
            "Pooled group-mean RMSE",
            "Pooled group-mean MAE",
        ]
    ].rename(
        columns={
            "Independent groups": "Within-specimen independent groups",
            "Pooled group-mean R2": "Within-specimen R2",
            "Pooled group-mean RMSE": "Within-specimen RMSE (kN)",
            "Pooled group-mean MAE": "Within-specimen MAE (kN)",
        }
    )


def version(name: str) -> str:
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return "unavailable"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = OUTPUT_ROOT / f"run_{stamp}_current0924"
    for folder in ("predictions", "tables", "diagnostics", "configs"):
        (run / folder).mkdir(parents=True, exist_ok=True)

    stage_frames = {}
    inventory_rows = []
    for stage in STAGES:
        frame, inventory = load_grouped_stage(stage)
        stage_frames[stage] = frame
        inventory_rows.extend(inventory)

    prediction_rows = []
    beam_metric_rows = []
    selection_rows = []
    candidate_rows = []

    for stage_index, (stage, stage_frame) in enumerate(stage_frames.items(), start=1):
        for feature_index, (feature_set, features) in enumerate(FEATURE_SETS.items(), start=1):
            for model_index, model in enumerate(MODELS, start=1):
                for heldout_beam in BEAMS:
                    print(
                        f"[{stage_index}/3] {stage} | [{feature_index}/{len(FEATURE_SETS)}] {feature_set} | "
                        f"[{model_index}/5] {model} | held-out Beam {heldout_beam}",
                        flush=True,
                    )
                    training = stage_frame[stage_frame["Beam"] != heldout_beam].reset_index(drop=True)
                    testing = stage_frame[stage_frame["Beam"] == heldout_beam].reset_index(drop=True)
                    best_cfg, inner_rmse, candidates = tune_on_training_beams(model, training, features)
                    _, prediction = fit_predict(model, best_cfg, training, testing, features)
                    baseline_value = float(training["Load"].mean())
                    baseline_prediction = np.full(len(testing), baseline_value)
                    score = metric_values(testing["Load"], prediction)
                    baseline_score = metric_values(testing["Load"], baseline_prediction)

                    selection_rows.append(
                        {
                            "Stage": stage,
                            "Feature set": feature_set,
                            "Model": model,
                            "Held-out Beam": heldout_beam,
                            "Training beams": "|".join(str(x) for x in BEAMS if x != heldout_beam),
                            "Inner mean held-out-beam RMSE (kN)": inner_rmse,
                            "Selected configuration": json.dumps(best_cfg, sort_keys=True),
                        }
                    )
                    for item in candidates:
                        candidate_rows.append(
                            {
                                "Stage": stage,
                                "Feature set": feature_set,
                                "Model": model,
                                "Outer held-out Beam": heldout_beam,
                                **item,
                            }
                        )

                    beam_metric_rows.append(
                        {
                            "Stage": stage,
                            "Feature set": feature_set,
                            "Model": model,
                            "Held-out Beam": heldout_beam,
                            "Independent test groups": len(testing),
                            **score,
                            "Training-mean baseline R2": baseline_score["R2"],
                            "Training-mean baseline RMSE (kN)": baseline_score["RMSE (kN)"],
                            "Training-mean baseline MAE (kN)": baseline_score["MAE (kN)"],
                            "RMSE improvement vs baseline (kN)": baseline_score["RMSE (kN)"] - score["RMSE (kN)"],
                        }
                    )
                    for row_index, row in testing.iterrows():
                        prediction_rows.append(
                            {
                                "Stage": stage,
                                "Feature set": feature_set,
                                "Model": model,
                                "Held-out Beam": heldout_beam,
                                "Group column": row["Group column"],
                                "Group value": row["Group value"],
                                "Technical replicates": int(row["technical_replicates"]),
                                "True Load": float(row["Load"]),
                                "Predicted Load": float(prediction[row_index]),
                                "Residual": float(prediction[row_index] - row["Load"]),
                                "Training-mean baseline prediction": baseline_value,
                            }
                        )

    predictions = pd.DataFrame(prediction_rows)
    beam_metrics = pd.DataFrame(beam_metric_rows)
    selections = pd.DataFrame(selection_rows)
    candidates = pd.DataFrame(candidate_rows)

    pooled_rows = []
    macro_rows = []
    for key, subset in predictions.groupby(["Stage", "Feature set", "Model"], sort=False):
        stage, feature_set, model = key
        score = metric_values(subset["True Load"], subset["Predicted Load"])
        baseline = metric_values(subset["True Load"], subset["Training-mean baseline prediction"])
        ci = cluster_bootstrap_ci(
            subset,
            seed=SEED + list(STAGES).index(stage) * 100 + list(FEATURE_SETS).index(feature_set) * 10 + list(MODELS).index(model),
        )
        pooled_rows.append(
            {
                "Stage": stage,
                "Feature set": feature_set,
                "Model": model,
                "Held-out beams": subset["Held-out Beam"].nunique(),
                "Independent test groups": len(subset),
                **score,
                **ci,
                "Training-mean baseline R2": baseline["R2"],
                "Training-mean baseline RMSE (kN)": baseline["RMSE (kN)"],
                "Training-mean baseline MAE (kN)": baseline["MAE (kN)"],
                "RMSE improvement vs baseline (kN)": baseline["RMSE (kN)"] - score["RMSE (kN)"],
            }
        )

        beam_subset = beam_metrics[
            (beam_metrics["Stage"] == stage)
            & (beam_metrics["Feature set"] == feature_set)
            & (beam_metrics["Model"] == model)
        ]
        macro_rows.append(
            {
                "Stage": stage,
                "Feature set": feature_set,
                "Model": model,
                "Beam-macro R2 mean": beam_subset["R2"].mean(),
                "Beam-macro R2 median": beam_subset["R2"].median(),
                "Beam-macro R2 SD": beam_subset["R2"].std(ddof=1),
                "Beam-macro RMSE mean (kN)": beam_subset["RMSE (kN)"].mean(),
                "Beam-macro RMSE median (kN)": beam_subset["RMSE (kN)"].median(),
                "Beam-macro RMSE SD (kN)": beam_subset["RMSE (kN)"].std(ddof=1),
                "Beam-macro MAE mean (kN)": beam_subset["MAE (kN)"].mean(),
                "Beam-macro MAE median (kN)": beam_subset["MAE (kN)"].median(),
                "Beam-macro MAE SD (kN)": beam_subset["MAE (kN)"].std(ddof=1),
                "Beams outperforming training-mean baseline": int(
                    (beam_subset["RMSE improvement vs baseline (kN)"] > 0).sum()
                ),
            }
        )

    pooled = pd.DataFrame(pooled_rows)
    macro = pd.DataFrame(macro_rows)
    within = read_within_specimen_metrics()
    lobo_full = pooled[pooled["Feature set"] == "Full15"].copy()
    comparison = within.merge(lobo_full, on=["Stage", "Model"], how="left", suffixes=("", " LOBO"))
    comparison["LOBO-to-within RMSE ratio"] = comparison["RMSE (kN)"] / comparison["Within-specimen RMSE (kN)"]
    comparison["LOBO minus within R2"] = comparison["R2"] - comparison["Within-specimen R2"]
    comparison["LOBO minus within MAE (kN)"] = comparison["MAE (kN)"] - comparison["Within-specimen MAE (kN)"]

    best_models = (
        pooled.sort_values(["Stage", "Feature set", "RMSE (kN)"])
        .groupby(["Stage", "Feature set"], as_index=False)
        .first()
    )

    inventory = pd.DataFrame(inventory_rows)
    predictions.to_csv(run / "predictions" / "lobo_group_mean_predictions.csv", index=False, encoding="utf-8-sig")
    beam_metrics.to_csv(run / "tables" / "lobo_metrics_by_beam.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(run / "tables" / "lobo_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(run / "tables" / "lobo_beam_macro_distribution.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(run / "tables" / "within_vs_lobo_comparison.csv", index=False, encoding="utf-8-sig")
    best_models.to_csv(run / "tables" / "best_models_by_stage.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(run / "configs" / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(run / "diagnostics" / "candidate_scores.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(run / "diagnostics" / "data_inventory.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "run": run.name,
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outer_validation": "Six folds; one complete beam held out in each fold",
        "inner_validation": "Leave-one-beam-out among the five outer-training beams",
        "analysis_unit": "Beam x physical-state group mean; technical replicates averaged before modeling",
        "hyperparameter_objective": "Equal-beam mean of inner held-out-beam RMSE",
        "models": list(MODELS),
        "feature_sets": FEATURE_SETS,
        "augmentation": False,
        "beam_as_feature": False,
        "bootstrap": f"{BOOTSTRAP_REPS} cluster resamples of the six held-out beams",
        "frequency_warning": (
            "Area1-Area12 are retained as originally defined within each beam-specific 60-kHz sweep "
            "(20-80 or 25-85 kHz); cross-beam LOBO results are therefore interpreted as exploratory."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "sklearn": version("sklearn"),
            "xgboost": version("xgboost"),
            "joblib": joblib.__version__,
        },
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "LATEST_RUN.txt").write_text(run.name + "\n", encoding="utf-8")

    print("\nBEST MODELS BY STAGE", flush=True)
    print(
        best_models[
            ["Stage", "Feature set", "Model", "R2", "RMSE (kN)", "MAE (kN)", "RMSE improvement vs baseline (kN)"]
        ].to_string(index=False),
        flush=True,
    )
    print(f"RUN={run}", flush=True)


if __name__ == "__main__":
    main()
