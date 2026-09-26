"""Randomized group-aware sensitivity analysis for the final 0924 data.

Short term: exact Load groups are randomly allocated to five outer folds.
Long term: complete Day groups are randomly allocated to five outer folds.

This is an interpolation-oriented sensitivity analysis. Random Day folds are
not a substitute for chronological validation because later dates may train a
model that predicts earlier dates.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
PROJECT = PACKAGE_ROOT / "03_outputs" / "_rerun_randomized"
WORKSPACE = PACKAGE_ROOT
base_path = HERE / "run_chronological_sensitivity.py"
spec = importlib.util.spec_from_file_location("group_aware_random_base", base_path)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.PROJECT = PROJECT
base.OUTPUT_ROOT = PROJECT
base.STAGES = {
    "short_term": {
        "folder": "1 短期加载", "file": "Database-Beam{beam}.xlsx",
        "group_col": "Load", "protocol": "nested_randomized_group_cv_load",
    },
    "long_term": {
        "folder": "2 长期监测", "file": "longterm-Beam{beam}.xlsx",
        "group_col": "Day", "protocol": "nested_randomized_group_cv_day",
    },
}


def random_outer(stage: str, groups):
    stage_offset = 0 if stage == "short_term" else 100
    return base.shuffled_group_folds(groups, 5, base.SEED + stage_offset)


def random_inner(stage: str, groups):
    stage_offset = 0 if stage == "short_term" else 100
    return base.shuffled_group_folds(groups, 3, base.SEED + 17 + stage_offset)


base.outer_folds = random_outer
base.inner_folds = random_inner


def main():
    base.main()
    run = Path((PROJECT / "LATEST_RUN.txt").read_text(encoding="utf-8").strip())
    pooled = pd.read_csv(run / "tables" / "pooled_group_metrics.csv")
    summary = pd.read_csv(run / "tables" / "nested_metrics_by_beam.csv")
    splits = pd.read_csv(run / "split_audit.csv")

    strict_project = PACKAGE_ROOT / "03_outputs" / "_rerun_chronological"
    strict_run = Path(
        (strict_project / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    )
    strict_pooled = pd.read_csv(strict_run / "tables" / "pooled_group_metrics.csv")
    strict_subset = strict_pooled[strict_pooled["Stage"].isin(base.STAGES)].copy()
    comparison = strict_subset.merge(
        pooled,
        on=["Stage", "Model"],
        suffixes=(" strict", " randomized-group"),
    )
    comparison.to_csv(run / "tables" / "comparison_with_strict.csv", index=False, encoding="utf-8-sig")

    lines = []
    for stage in base.STAGES:
        ranked = pooled[pooled["Stage"] == stage].sort_values("Pooled group-mean RMSE")
        hit = ranked.iloc[0]
        lines.append(
            f"- {stage}: {hit['Model']} has the lowest descriptive pooled group RMSE, "
            f"R² = {hit['Pooled group-mean R2']:.4f}, RMSE = {hit['Pooled group-mean RMSE']:.4f} kN, "
            f"MAE = {hit['Pooled group-mean MAE']:.4f} kN."
        )

    report = f"""# Randomized group-aware sensitivity analysis

## Decision

- Short-term loading has no repeated loading cycles. Cycle holdout is not applicable.
- Short-term records are grouped by exact Load and randomly assigned to five outer folds.
- Long-term records are grouped by Day and randomly assigned to five outer folds.
- Every technical replicate from one Load or Day remains in one fold.
- Hyperparameters are selected by three-fold grouped validation inside each outer training set.
- No augmentation is used.
- Primary metrics use the mean prediction of each independent group.
- Confidence intervals use 2,000 independent-group bootstrap resamples.

## Leakage audit

- Outer split records: {len(splits)}.
- Maximum train/test group overlap: {int(splits['Train-test group overlap'].max())}.
- Groups tested exactly once: yes.

## Descriptive randomized-group results

{chr(10).join(lines)}

## Interpretation boundary

Randomized short-term Load-group validation estimates interpolation within the
single observed monotonic loading trajectory. It does not establish
generalization to a new loading cycle.

Randomized long-term Day-group validation prevents same-day replicate leakage,
but it is not time-causal: later dates may appear in training when an earlier
date is tested. It should therefore be reported as a sensitivity analysis beside,
not instead of, rolling-origin validation.

The loading-to-failure analysis was not rerun here because the strict run already
uses randomized Applied-load group folds. Its result is unchanged.
"""
    (run / "random_group_report.md").write_text(report, encoding="utf-8")

    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["analysis_role"] = "randomized group-aware sensitivity analysis"
    config["failure_stage"] = "not rerun; existing strict run already uses randomized Applied-load groups"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RANDOM_GROUP_RUN={run.resolve()}", flush=True)


if __name__ == "__main__":
    main()
