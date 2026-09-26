"""Leakage-safe nested validation for the final 0924 EMI datasets.

Protocols
---------
short_term: no repeated loading cycles exist. Exact target loads are kept
             together and contiguous load ranges are held out.
long_term:  monitoring Day is the group; rolling-origin validation trains on
             earlier dates and tests the next date.
failure:    Applied load is the group; complete loading stages are held out.

No augmentation is used. Hyperparameters are selected only within each outer
training partition. Primary metrics are calculated after averaging predictions
within each independent group, so technical replicates do not inflate n.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
PROJECT = PACKAGE_ROOT / "03_outputs" / "_rerun_chronological"
WORKSPACE = PACKAGE_ROOT
DATA_ROOT = PACKAGE_ROOT / "01_data"
OUTPUT_ROOT = PROJECT
SEED = 42
BOOTSTRAP_REPS = 2000
FEATURES = ["RMSD", "Peak", "Frequency"] + [f"Area{i}" for i in range(1, 13)]
TARGET = "Load"
MODELS = ("DT", "SVM", "RF", "MLP", "XGBoost")
BEAMS = range(1, 7)
STAGES = {
    "short_term": {
        "folder": "1 短期加载", "file": "Database-Beam{beam}.xlsx",
        "group_col": "Load", "protocol": "nested_contiguous_load_block",
    },
    "long_term": {
        "folder": "2 长期监测", "file": "longterm-Beam{beam}.xlsx",
        "group_col": "Day", "protocol": "nested_rolling_origin_day",
    },
    "failure": {
        "folder": "3 加载破坏", "file": "failure-Beam{beam}.xlsx",
        "group_col": "Applied load", "protocol": "nested_grouped_applied_load",
    },
}

helper_path = HERE / "model_space.py"
spec = importlib.util.spec_from_file_location("final0924_model_space", helper_path)
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


def source_path(stage: str, beam: int) -> Path:
    cfg = STAGES[stage]
    return DATA_ROOT / cfg["folder"] / cfg["file"].format(beam=beam)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_data(stage: str, beam: int) -> pd.DataFrame:
    path = source_path(stage, beam)
    raw = pd.read_excel(path)
    group_col = STAGES[stage]["group_col"]
    needed = FEATURES + [TARGET]
    if group_col != TARGET:
        needed.append(group_col)
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    data = raw[needed].apply(pd.to_numeric, errors="coerce")
    if data.isna().any().any():
        bad = data.index[data.isna().any(axis=1)].tolist()
        raise ValueError(f"{path}: missing/non-numeric values at source rows {bad}")
    if not np.isfinite(data.to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite values")
    data = data.reset_index(names="Source row")
    data["Source row"] += 2
    conflicts = data.groupby(group_col)[TARGET].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError(f"{path}: target conflict inside groups {conflicts[conflicts > 1].to_dict()}")
    return data


def indices_for(groups: np.ndarray, test_groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(np.isin(groups, test_groups))
    train = np.flatnonzero(~np.isin(groups, test_groups))
    return train, test


def contiguous_group_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups.astype(float))
    n_splits = min(int(n_splits), len(unique))
    if n_splits < 2:
        raise ValueError("At least two groups are required")
    blocks = [b for b in np.array_split(np.sort(unique), n_splits) if len(b)]
    return [indices_for(groups, block) for block in blocks]


def shuffled_group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    n_splits = min(int(n_splits), len(unique))
    if n_splits < 2:
        raise ValueError("At least two groups are required")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    blocks = [b for b in np.array_split(shuffled, n_splits) if len(b)]
    return [indices_for(groups, block) for block in blocks]


def rolling_origin_folds(groups: np.ndarray, min_train_groups: int, max_folds: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.sort(np.unique(groups.astype(float)))
    if len(unique) <= min_train_groups:
        raise ValueError(f"Need more than {min_train_groups} chronological groups")
    starts = list(range(min_train_groups, len(unique)))
    if max_folds is not None:
        starts = starts[-max_folds:]
    folds = []
    for i in starts:
        train_groups = unique[:i]
        test_groups = unique[i:i + 1]
        train = np.flatnonzero(np.isin(groups, train_groups))
        test = np.flatnonzero(np.isin(groups, test_groups))
        folds.append((train, test))
    return folds


def outer_folds(stage: str, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if stage == "short_term":
        return contiguous_group_folds(groups, 4)
    if stage == "failure":
        return shuffled_group_folds(groups, 4, SEED)
    return rolling_origin_folds(groups, min_train_groups=5)


def inner_folds(stage: str, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if stage == "short_term":
        return contiguous_group_folds(groups, 3)
    if stage == "failure":
        return shuffled_group_folds(groups, 3, SEED + 17)
    return rolling_origin_folds(groups, min_train_groups=3, max_folds=3)


def fit_predict(estimator, X_train, y_train, X_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(X_train, y_train)
    prediction = np.asarray(estimator.predict(X_test), float)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    return prediction


def metric_values(y, pred) -> dict[str, float]:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return {
        "R2": float(r2_score(y, pred)) if len(y) > 1 and np.unique(y).size > 1 else np.nan,
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
    }


def group_frame(y, pred, groups) -> pd.DataFrame:
    return (
        pd.DataFrame({"True Load": y, "Predicted Load": pred, "Group": groups})
        .groupby("Group", as_index=False)
        .agg({"True Load": "mean", "Predicted Load": "mean"})
    )


def group_metrics(y, pred, groups) -> dict[str, float]:
    frame = group_frame(y, pred, groups)
    return metric_values(frame["True Load"], frame["Predicted Load"])


def tune(stage: str, model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    folds = inner_folds(stage, groups)
    best_cfg = None
    best_rmse = np.inf
    for cfg in helper.candidates(model):
        fold_records = []
        failed = False
        for train, test in folds:
            if len(train) == 0 or len(test) == 0:
                failed = True
                break
            try:
                estimator = helper.make_estimator(model, cfg)
                prediction = fit_predict(estimator, X[train], y[train], X[test])
                fold_records.append(pd.DataFrame({
                    "y": y[test], "pred": prediction, "group": groups[test],
                }))
            except Exception:
                failed = True
                break
        if failed or not fold_records:
            continue
        combined = pd.concat(fold_records, ignore_index=True)
        score = group_metrics(combined["y"], combined["pred"], combined["group"])["RMSE"]
        if score < best_rmse:
            best_rmse = float(score)
            best_cfg = cfg.copy()
    if best_cfg is None:
        raise RuntimeError(f"No valid configuration: {stage} {model}")
    return best_cfg, best_rmse


def bootstrap_ci(grouped: pd.DataFrame, seed: int, reps: int = BOOTSTRAP_REPS) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(grouped)
    values = {"R2": [], "RMSE": [], "MAE": []}
    for _ in range(reps):
        sampled = grouped.iloc[rng.integers(0, n, size=n)]
        score = metric_values(sampled["True Load"], sampled["Predicted Load"])
        for key, value in score.items():
            if np.isfinite(value):
                values[key].append(value)
    result = {}
    for key in ("R2", "RMSE", "MAE"):
        if values[key]:
            result[f"{key} CI lower"] = float(np.percentile(values[key], 2.5))
            result[f"{key} CI upper"] = float(np.percentile(values[key], 97.5))
        else:
            result[f"{key} CI lower"] = np.nan
            result[f"{key} CI upper"] = np.nan
    return result


def fmt_group_values(values) -> str:
    return "|".join(f"{float(x):g}" for x in np.sort(np.unique(values)))


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = OUTPUT_ROOT / f"run_{stamp}_final0924"
    for folder in ("diagnostics", "predictions", "models", "tables"):
        (run / folder).mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    split_rows = []
    fold_metric_rows = []
    summary_rows = []
    prediction_rows = []
    hyper_rows = []

    for stage in STAGES:
        group_col = STAGES[stage]["group_col"]
        for beam in BEAMS:
            data = load_data(stage, beam)
            X = data[FEATURES].to_numpy(float)
            y = data[TARGET].to_numpy(float)
            groups = data[group_col].to_numpy()
            group_sizes = data.groupby(group_col).size()
            outer = outer_folds(stage, groups)
            tested = np.zeros(len(data), dtype=int)

            inventory_rows.append({
                "Stage": stage,
                "Beam": beam,
                "File": str(source_path(stage, beam).resolve()),
                "SHA256": sha256(source_path(stage, beam)),
                "Rows": len(data),
                "Group column": group_col,
                "Independent groups": int(data[group_col].nunique()),
                "Minimum rows per group": int(group_sizes.min()),
                "Maximum rows per group": int(group_sizes.max()),
                "Groups with one row": int((group_sizes == 1).sum()),
                "Exact duplicate feature-target rows": int(data.duplicated(FEATURES + [TARGET]).sum()),
                "Target conflicts within group": 0,
                "Protocol": STAGES[stage]["protocol"],
            })

            for fold_id, (train, test) in enumerate(outer, start=1):
                tested[test] += 1
                train_groups = np.unique(groups[train])
                test_groups = np.unique(groups[test])
                overlap = np.intersect1d(train_groups, test_groups)
                chronology_ok = True
                if stage == "long_term" and "rolling_origin" in STAGES[stage]["protocol"]:
                    chronology_ok = float(np.max(train_groups)) < float(np.min(test_groups))
                if len(overlap) or not chronology_ok:
                    raise RuntimeError(f"Invalid split: {stage} Beam {beam} fold {fold_id}")
                split_rows.append({
                    "Stage": stage, "Beam": beam, "Outer fold": fold_id,
                    "Protocol": STAGES[stage]["protocol"], "Group column": group_col,
                    "Training rows": len(train), "Testing rows": len(test),
                    "Training groups": len(train_groups), "Testing groups": len(test_groups),
                    "Train-test group overlap": len(overlap),
                    "Chronology respected": chronology_ok,
                    "Training group values": fmt_group_values(train_groups),
                    "Testing group values": fmt_group_values(test_groups),
                })

            if stage == "long_term" and "rolling_origin" in STAGES[stage]["protocol"]:
                if np.any(tested > 1) or np.any(tested[np.isin(groups, np.sort(np.unique(groups))[5:])] != 1):
                    raise RuntimeError(f"Incomplete rolling-origin coverage: {stage} Beam {beam}")
            elif not np.all(tested == 1):
                raise RuntimeError(f"Incomplete grouped OOF coverage: {stage} Beam {beam}")

            for model_id, model in enumerate(MODELS):
                print(f"START {stage} Beam{beam} {model}", flush=True)
                oof = np.full(len(data), np.nan)
                fold_assignment = np.zeros(len(data), dtype=int)

                for fold_id, (train, test) in enumerate(outer, start=1):
                    cfg, inner_rmse = tune(stage, model, X[train], y[train], groups[train])
                    estimator = helper.make_estimator(model, cfg)
                    pred = fit_predict(estimator, X[train], y[train], X[test])
                    oof[test] = pred
                    fold_assignment[test] = fold_id
                    row_score = metric_values(y[test], pred)
                    primary = group_metrics(y[test], pred, groups[test])
                    fold_metric_rows.append({
                        "Stage": stage, "Beam": beam, "Model": model, "Outer fold": fold_id,
                        "Test rows": len(test), "Test groups": len(np.unique(groups[test])),
                        "Inner selected group RMSE": inner_rmse,
                        **{f"Row {k}": v for k, v in row_score.items()},
                        **{f"Group-mean {k}": v for k, v in primary.items()},
                        "Best hyperparameters": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                    })

                evaluated = np.flatnonzero(np.isfinite(oof))
                if len(evaluated) == 0 or np.any(fold_assignment[evaluated] == 0):
                    raise RuntimeError(f"Missing OOF predictions: {stage} Beam {beam} {model}")
                row_score = metric_values(y[evaluated], oof[evaluated])
                grouped = group_frame(y[evaluated], oof[evaluated], groups[evaluated])
                primary = metric_values(grouped["True Load"], grouped["Predicted Load"])
                ci = bootstrap_ci(grouped, SEED + beam * 101 + model_id * 1009)
                fold_subset = [r for r in fold_metric_rows if r["Stage"] == stage and r["Beam"] == beam and r["Model"] == model]
                fold_rmses = np.array([r["Group-mean RMSE"] for r in fold_subset], float)

                final_cfg, full_inner_rmse = tune(stage, model, X, y, groups)
                final_estimator = helper.make_estimator(model, final_cfg)
                full_fit_pred = fit_predict(final_estimator, X, y, X)
                fit_score = group_metrics(y, full_fit_pred, groups)
                model_dir = run / "models" / stage
                model_dir.mkdir(exist_ok=True)
                joblib.dump(final_estimator, model_dir / f"Beam{beam}_{model}.joblib")
                hyper_rows.append({
                    "Stage": stage, "Beam": beam, "Model": model,
                    "Full-data inner group RMSE": full_inner_rmse,
                    "Final hyperparameters": json.dumps(final_cfg, ensure_ascii=False, sort_keys=True),
                })

                summary_rows.append({
                    "Stage": stage, "Beam": beam, "Model": model,
                    "Protocol": STAGES[stage]["protocol"],
                    "Total rows": len(data), "Total independent groups": data[group_col].nunique(),
                    "OOF evaluated rows": len(evaluated), "OOF evaluated groups": len(grouped),
                    **{f"Primary group-mean {k}": v for k, v in primary.items()},
                    **ci,
                    **{f"Secondary row {k}": v for k, v in row_score.items()},
                    "Fold RMSE median": float(np.median(fold_rmses)),
                    "Fold RMSE Q1": float(np.percentile(fold_rmses, 25)),
                    "Fold RMSE Q3": float(np.percentile(fold_rmses, 75)),
                    **{f"Full-data fit group-mean {k}": v for k, v in fit_score.items()},
                })

                for idx in evaluated:
                    prediction_rows.append({
                        "Stage": stage, "Beam": beam, "Model": model,
                        "Source Excel row": int(data.loc[idx, "Source row"]),
                        "Outer fold": int(fold_assignment[idx]),
                        "Group column": group_col, "Group value": groups[idx],
                        "True Load": y[idx], "Predicted Load": oof[idx],
                        "Residual": oof[idx] - y[idx],
                        "Prediction type": "nested out-of-fold",
                    })
                print(f"DONE  {stage} Beam{beam} {model} primary_RMSE={primary['RMSE']:.6f}", flush=True)

    inventory = pd.DataFrame(inventory_rows)
    splits = pd.DataFrame(split_rows)
    folds = pd.DataFrame(fold_metric_rows)
    summary = pd.DataFrame(summary_rows)
    predictions = pd.DataFrame(prediction_rows)
    hypers = pd.DataFrame(hyper_rows)

    pooled_rows = []
    for (stage, model), subset in predictions.groupby(["Stage", "Model"]):
        grouped = (
            subset.assign(IndependentGroup=subset.apply(lambda r: f"Beam{int(r['Beam'])}|{r['Group value']}", axis=1))
            .groupby("IndependentGroup", as_index=False)
            .agg({"True Load": "mean", "Predicted Load": "mean"})
        )
        score = metric_values(grouped["True Load"], grouped["Predicted Load"])
        ci = bootstrap_ci(grouped, SEED + list(STAGES).index(stage) * 10000 + list(MODELS).index(model) * 1000)
        beam_subset = summary[(summary["Stage"] == stage) & (summary["Model"] == model)]
        pooled_rows.append({
            "Stage": stage, "Model": model, "Independent groups": len(grouped),
            **{f"Pooled group-mean {k}": v for k, v in score.items()},
            **ci,
            "Beam-macro R2 mean": beam_subset["Primary group-mean R2"].mean(),
            "Beam-macro RMSE mean": beam_subset["Primary group-mean RMSE"].mean(),
            "Beam-macro MAE mean": beam_subset["Primary group-mean MAE"].mean(),
        })
    pooled = pd.DataFrame(pooled_rows)

    inventory.to_csv(run / "data_inventory.csv", index=False, encoding="utf-8-sig")
    splits.to_csv(run / "split_audit.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(run / "diagnostics" / "nested_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(run / "tables" / "nested_metrics_by_beam.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(run / "tables" / "pooled_group_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(run / "predictions" / "nested_oof_predictions.csv", index=False, encoding="utf-8-sig")
    hypers.to_csv(run / "tables" / "final_hyperparameters.csv", index=False, encoding="utf-8-sig")

    best_lines = []
    for stage in STAGES:
        hit = pooled[pooled["Stage"] == stage].sort_values("Pooled group-mean RMSE").iloc[0]
        best_lines.append(
            f"- {stage}: lowest descriptive pooled group RMSE = {hit['Pooled group-mean RMSE']:.4f} kN "
            f"({hit['Model']}); R² = {hit['Pooled group-mean R2']:.4f}, MAE = {hit['Pooled group-mean MAE']:.4f} kN."
        )

    report = """# Final 0924 group-aware training and prediction

## Validation decision

- Adding Day and Applied load makes leakage-safe splitting possible, but the labels alone do not prevent leakage; the splitter must consume them.
- Short-term loading has no repeated cycles. Complete-cycle holdout is therefore not applicable. Equal Load values are kept together, and contiguous load ranges are held out.
- Long-term monitoring is grouped by Day and evaluated by rolling origin: earlier dates train the model and the next date is tested.
- Loading-to-failure is grouped by Applied load; all repetitions from one specimen-stage remain in one fold.
- Hyperparameter selection is nested inside each outer training partition. No augmentation is used.
- Primary metrics use one averaged prediction per independent group. Row-level metrics are secondary diagnostics.
- Confidence intervals use 2,000 bootstrap resamples of independent groups, not individual technical replicates.

## Independent groups

Long-term monitoring has 11 dates per beam. Loading-to-failure has 11 stages for Beams 1–4 and 6, and 9 stages for Beam 5. Short-term independent groups are the distinct Load values; no cycle-level generalization claim is made.

## Descriptive lowest pooled RMSE by stage

""" + "\n".join(best_lines) + """

These model comparisons are descriptive outer-validation results. They were not fed back into hyperparameter fitting. Negative R² values, if present, are retained.

## Output map

- `data_inventory.csv`: row counts, independent group counts, duplicates and source hashes.
- `split_audit.csv`: every outer fold and the zero-overlap/chronology checks.
- `diagnostics/nested_metrics_by_fold.csv`: performance distribution across outer folds.
- `tables/nested_metrics_by_beam.csv`: group-level primary metrics and 95% bootstrap intervals.
- `tables/pooled_group_metrics.csv`: pooled independent-group results by stage and model.
- `predictions/nested_oof_predictions.csv`: leakage-safe predictions for every evaluated row.
- `tables/final_hyperparameters.csv` and `models/`: full-data fits for future use; their fit metrics are not validation estimates.
"""
    (run / "validation_report.md").write_text(report, encoding="utf-8")
    (run / "run_config.json").write_text(json.dumps({
        "seed": SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "features": FEATURES,
        "target": TARGET,
        "models": MODELS,
        "augmentation": "none",
        "protocols": {k: {"group_column": v["group_col"], "protocol": v["protocol"]} for k, v in STAGES.items()},
        "long_term_warmup_dates": 5 if "long_term" in STAGES and "rolling_origin" in STAGES["long_term"]["protocol"] else None,
        "primary_unit": "independent group mean",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROJECT / "LATEST_RUN.txt").write_text(str(run.resolve()), encoding="utf-8")
    print(f"RUN_DIR={run.resolve()}", flush=True)


if __name__ == "__main__":
    main()
