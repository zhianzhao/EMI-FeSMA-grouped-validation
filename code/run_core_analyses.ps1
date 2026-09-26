$ErrorActionPreference = "Stop"
$codeRoot = $PSScriptRoot

python "$codeRoot\run_chronological_sensitivity.py"
python "$codeRoot\run_randomized_group_primary.py"
python "$codeRoot\build_long_term_comparison.py"
python "$codeRoot\run_lobo_current.py"
python "$codeRoot\generate_shap_beam2.py"

Write-Host "Core analyses completed. Results are in 03_outputs."
