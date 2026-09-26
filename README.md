# EMI-FeSMA-grouped-validation

Core analysis code and reproducibility records for EMI-based force evaluation
of unbonded Fe-SMA rebars using PZT smart washers.

## Included materials

- `code/`: grouped nested validation, rolling-origin validation, LOBO
  validation, and the Beam 2 XGBoost TreeSHAP analysis.
- `Table_S1_data_and_group_counts.csv`: record and physical-state group counts.
- `Table_S2_fold_specific_configurations.csv`: configurations selected within
  the outer folds.
- `run_config.json`: common features, seeds, fold definitions, and bootstrap
  settings.

No synthetic data augmentation is used. Technical replicates belonging to the
same physical state are kept in the same partition, and reported performance
is calculated from held-out group-mean predictions.

## Expected data layout

The raw experimental files are not included in this repository. To reproduce
the analyses, place the files supplied by the corresponding author under:

```text
01_data/
  1 短期加载/Database-Beam1.xlsx ... Database-Beam6.xlsx
  2 长期监测/longterm-Beam1.xlsx ... longterm-Beam6.xlsx
  3 加载破坏/failure-Beam1.xlsx ... failure-Beam6.xlsx
```

Each workbook must contain `RMSD`, `Peak`, `Frequency`, `Area1` to `Area12`,
and `Load`. Long-term files additionally require `Day`; loading-to-failure
files require `Applied load`.

## Environment and execution

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File code/run_core_analyses.ps1
```

The scripts create their outputs under `03_outputs/`. The base random seed is
42. Bootstrap confidence intervals use 2,000 resamples for beam-specific and
rolling-origin analyses and 3,000 beam-cluster resamples for LOBO validation.

## Data availability

The supporting experimental data and group identifiers are available from the
corresponding author upon reasonable request.
