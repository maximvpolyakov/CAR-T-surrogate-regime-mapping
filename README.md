# CAR-T surrogate-assisted sensitivity analysis and response-regime mapping

This repository contains the Python pipeline and C++ PDE solver used for adaptive surrogate-assisted global sensitivity analysis and four-class response-regime mapping in a spatiotemporal CAR-T therapy model.

## Requirements and execution

- Windows with Python 3.10 or newer; a compatible C++17 compiler is needed for building the PDE solver. OpenMP support is optional.
- Python dependencies are listed in `pyproject.toml`.
- Model coefficients, numerical settings, random seeds, sampling budgets, response thresholds, and validation settings are specified in `configs/study.yaml`.

On Windows, run `run_study.bat` from the repository root. This script creates a Python environment, installs any missing Python dependencies, and launches the main pipeline using:

```text
python -m cart_pipeline --config configs/study.yaml --output results
```

For a manual Python setup, run:

```text
python -m pip install -e .
cart-study --config configs/study.yaml --output results
```

**Computational cost:** a complete execution includes many full three-dimensional PDE simulations and is not a quick smoke test. If results have already been generated, use `run_replot.bat` to regenerate supported figures from the `results/` directory without repeating the PDE simulations.

## Published CSV datasets

The repository root currently contains two PDE-generated CSV datasets:

| File | Contents | Role |
| --- | --- | --- |
| `training_data.csv` | 120 configurations: 48 initial LHS points and 72 adaptively selected points | Training the final adaptive GP surrogates |
| `validation_data.csv` | 30 independently selected configurations | Validation of surrogate predictions and stopping diagnostics during sequential sampling |

Each row corresponds to a parameter configuration. The main columns are:

- `s_mig`, `s_cyt`, `s_exh`, `s_esc`, `s_env`, `s_hyp`: six dimensionless mechanism multipliers in the interval [0.6, 1.4].
- `z_s_mig`, `z_s_cyt`, `z_s_exh`, `z_s_esc`, `z_s_env`, `z_s_hyp`: the same variables linearly scaled to [0, 1].
- `Y_end`: final normalised tumour burden.
- `Y_min`: minimum normalised tumour burden over the simulated horizon.
- `Y_reb`: an auxiliary PDE-generated response quantity saved by the solver.
- `regime`: PDE-based response class (`DC`, `PR`, `RL`, or `Failure`).
- `stage` (training set): `initial` or `adaptive`, indicating how the training point was obtained.
- `cache_key`: identifier used for matching an individual PDE evaluation to the cached simulation output.

The six multiplier columns are the inputs of the original PDE simulations; their normalised versions are supplied to the Gaussian-process surrogates. The two primary surrogate targets are `Y_end` and `Y_min`. The `regime` column is derived from these two outputs using the thresholds in `configs/study.yaml`.

**Important distinction:** the root-level CSVs are archived *results of PDE evaluations*. The standard command `run_study.bat` generates its own experiment using the selected configuration; it does not automatically read the root-level CSVs as a replacement for PDE computations.

## Validation protocol

The code uses two distinct 30-point datasets for different purposes.

### Independent validation during adaptive sampling

The initial validation design contains 30 maximin-LHS configurations generated with a seed distinct from the training design. These points are evaluated with the full PDE model and are not appended to the GP training set. Their predictions are used at diagnostic checkpoints to calculate NRMSE and regime-classification accuracy. Those metrics contribute to the adaptive stopping decision.

The relevant implementation is in `src/cart_pipeline/adaptive.py` (`prepare_validation` and `_checkpoint`) and `src/cart_pipeline/validation.py` (`regression_metrics` and `validate_surrogate`). The primary outputs of a complete pipeline run are:

- `results/data/adaptive/validation_data.csv`: PDE outputs for this validation design;
- `results/data/validation_predictions.csv`: PDE outputs and predictions from the final adaptive and space-filling GP surrogates;
- `results/data/validation_metrics.json`: validation errors and classification diagnostics for both designs;
- `results/data/checkpoint_metrics_long.csv`: metrics tracked as the training budget increases.

The NRMSE is calculated as `RMSE / s_Y`, where `s_Y` is the sample standard deviation (`ddof=1`) of the corresponding PDE response in the evaluation dataset. Regime accuracy is the fraction of configurations for which the predicted and PDE-derived classes agree. The program also records R-squared, MAE, maximum absolute error, balanced classification accuracy, and a confusion matrix.

### Independent blind test after training

After the sampling and regime-map procedures, a *different* 30-point maximin-LHS design is created (`blind_test` settings in `configs/study.yaml`). The software checks that the blind points do not coincide with the adaptive training set, space-filling training set, earlier validation set, selected regime-boundary checks, or one-factor analysis points. The blind dataset is reserved for final evaluation; it is not used for acquisition or the stopping decision.

The implementation is in `src/cart_pipeline/blind.py`. A complete run writes:

- `results/data/blind_test/blind_design_reserved.csv`: blind-test input coordinates;
- `results/data/blind_test/blind_test_predictions.csv`: full PDE outputs, GP predictions, and linear/quadratic regression predictions;
- `results/data/blind_test/blind_test_metrics.json`: blind-test regression, classification, and GP interval-coverage metrics;
- `results/data/blind_test/blind_model_comparison.csv`: summary of surrogate baselines.

These blind-test files are **generated by the pipeline** and are not the same as the root-level `validation_data.csv`.

### Direct PDE verification near response boundaries

The pipeline selects additional configurations near boundaries in the pairwise GP response-regime maps and reevaluates them with the full PDE solver. The corresponding file is `results/data/regime_boundary_checks.csv`, including parameter values, predicted and PDE regimes, predictions, PDE outputs, and classification agreement. This check is distinct from both the stopping validation set and the final blind test.

## Output structure

A complete study writes to the chosen output directory (`results/` by default):

```text
results/
  config_used.yaml
  environment.json
  data/
    adaptive/
      training_data.csv
      validation_data.csv
      adaptive_checkpoints.json
      acquisition_history.csv
    comparator/
    validation_predictions.csv
    validation_metrics.json
    checkpoint_metrics_long.csv
    blind_test/
      blind_design_reserved.csv
      blind_test_predictions.csv
      blind_test_metrics.json
      blind_model_comparison.csv
    regime_boundary_checks.csv
    sobol_indices_final.csv
    regime_maps/
  reports/
    summary.json
    computational_budget.json
  models/
  figures/
```

The software also records `config_used.yaml` and `environment.json` to document the effective settings and software environment of each run.

## Numerical reproducibility

Before comparing output values against the manuscript, verify that the numerical settings in the YAML file correspond to the configuration used for the reported calculations. In particular, the spatial grid size (`pde.grid.nx`, `ny`, and `nz`), timestep (`pde.time.dt`), random seed (`study.seed`), and GP fitting settings must match. Numerical results may change when solver settings or software versions change.

To enable independent verification *without* rerunning the computationally intensive PDE study, the corresponding archived configuration and the final blind-test, comparison, and boundary-check datasets should be published alongside the code.
