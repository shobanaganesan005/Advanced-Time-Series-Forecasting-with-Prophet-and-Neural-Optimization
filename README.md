# Advanced Time Series Forecasting with Prophet and Neural Optimization

This repository provides a clean, modular implementation of the assignment:

- Load or generate a **complex time series** with clear trend/seasonality.
- Implement a **SARIMAX baseline** and record performance across several horizons.
- Implement **Facebook Prophet** and wrap it in an objective function.
- Use a **metaheuristic optimizer** (SciPy Differential Evolution) as the
  "neural optimization" component to tune Prophet hyperparameters.
- Compare the optimised Prophet model against SARIMAX.

`project_description_350.txt` contains a 350-character summary that you can
paste into short description fields.

## Structure

- `src/data_generator.py` – load AirPassengers from statsmodels or simulate a custom series.
- `src/evaluate.py` – MAE, RMSE, MAPE utilities and helper functions.
- `src/baseline_sarimax.py` – SARIMAX training and evaluation.
- `src/prophet_model.py` – Prophet training/evaluation wrappers.
- `src/optimize_prophet.py` – differential-evolution search over Prophet hyperparameters.
- `src/pipeline.py` – end-to-end run: data → baseline → Prophet default → optimised Prophet.
- `data/air_passengers.csv` – small built-in dataset (can be swapped with your own).
- `outputs/` – metrics JSON/CSV and optimisation logs.
- `report/report.md` – template for the written analysis.

## Quick Start

```bash
pip install -r requirements.txt
python src/data_generator.py --use-airpassengers --output data/air_passengers.csv
python src/pipeline.py --data-path data/air_passengers.csv
```

This will:

1. Fit a SARIMAX baseline and evaluate MAE/RMSE/MAPE for horizons [6, 12, 24].
2. Fit a default Prophet model and compute the same metrics.
3. Run differential evolution to optimise Prophet hyperparameters.
4. Refit Prophet with the best settings and re-evaluate.
5. Save comparison tables into `outputs/metrics_summary.csv` and JSON files.
