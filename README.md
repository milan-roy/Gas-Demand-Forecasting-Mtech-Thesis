# Gas Demand Forecasting — MTech Thesis (IIT Kanpur)

Forecasting daily industrial gas demand over a 30-day horizon using deep learning, developed as part of an MTech thesis at IIT Kanpur.

---

## Project Overview

- **Task:** Multi-output regression — given 30 days of historical demand, predict the next 30 days
- **Data period:** April 1, 2020 – December 31, 2024 (daily, in SCMD — Standard Cubic Meters per Day)
- **Models:** Feed-Forward Neural Network (FFNN), GRU, Transformer (all implemented in PyTorch)
- **Loss function:** Custom Asymmetric MSE (AMSE) with tunable under/over-prediction penalty weights
- **Tuning:** Optuna — single-objective (minimise MAPE) and multi-objective (jointly minimise MAPE + MUPE)

---

## Data

Place the following files inside the `Data/` directory before running any notebooks or scripts.

| File | Columns | Description |
|---|---|---|
| `DemandData.xlsx` | `Date`, `Sales` | Raw daily industrial gas demand (SCMD). One row per day. |
| `CrudeOilPrice.xlsx` | `Date`, `Value` | Daily crude oil prices. Weekends/holidays filled via LOCF. |
| `holidays.csv` | — | Holiday calendar with pre/post effect windows (provided). |

The following files are **generated automatically** by `denoisingDataset.py` and should not be edited manually:

| File | Description |
|---|---|
| `DataMissingValuesFilled.xlsx` | Raw data after interpolating the April 2020 – April 2022 gap |
| `FixedData_Random_Smoothed.xlsx` | Holiday-adjusted dataset (hybrid: 50% random uniform + 50% linear smoothing within holiday windows) |

> **Holiday calendar:** 8 major Indian holidays with manually calibrated pre/post effect windows (e.g. Diwali: −3 days / +12 days; Republic Day/Independence Day: 0 days either side).

---

## Repository Structure

```
Gas-Demand-Forecasting-Mtech-Thesis/
│
├── Data/                              # Input data (see above)
│
├── HyperParameterTuning/              # Single-objective Optuna tuning (minimise MAPE)
│   ├── FFNN.ipynb
│   ├── GRU.ipynb
│   └── Transformer.ipynb
│
├── Evaluation-MultipleRuns/           # Run best hyperparameters N times, aggregate metrics
│   ├── FFNN.ipynb
│   ├── GRU.ipynb
│   └── Transformer.ipynb
│
├── Multi Objective Optimization/      # Multi-objective tuning (minimise MAPE + MUPE jointly)
│   ├── FFNN.ipynb                     # Optuna NSGAii search → Pareto front per config
│   ├── best_hyperparameters_weighted.py  # Ranks Pareto front, returns top-5 per config
│   ├── MultipleRuns.ipynb             # Evaluates top-5 ranked params over N runs
│   └── print_best_config_table.py    # Applies weighted scoring to MultipleRuns results
│
├── Training and Inference/            # Train best params, run inference, generate plots/tables
│   ├── FFNN.ipynb
│   ├── GRU.ipynb
│   └── Transfomer.ipynb
│
├── dataset.py                         # GasDemandDataset — sliding-window sequence builder
├── denoisingDataset.py                # Missing value imputation + holiday-adjusted dataset
├── engine.py                          # AMSE loss function, training loop, prediction engine
├── holidayCorrection.py               # Post-processing holiday correction for raw predictions
├── Models.py                          # FFNN, GRU, Transformer model definitions (PyTorch)
└── utils.py                           # Data loading/prep, metric calculation, Plotly plots
```

---

## Core Modules

### `dataset.py`
Contains `GasDemandDataset` — a PyTorch `Dataset` that loads the processed data and constructs sliding-window input/output sequences (30-day input → 30-day forecast). Handles optional inclusion of crude oil price and calendar features.

### `denoisingDataset.py`
Two responsibilities:
1. Fills the missing-value gap (April 4, 2020 – April 7, 2022) via interpolation and saves `DataMissingValuesFilled.xlsx`.
2. Identifies holiday-affected windows using `holidays.csv` and replaces them with a hybrid interpolation (50% random uniform + 50% linear smoothing between boundary values), saving `FixedData_Random_Smoothed.xlsx`.

### `engine.py`
- **AMSE loss:** Asymmetric MSE with separate `under_parameter` (α) and `over_parameter` (β) weights applied to normalised prediction error.
- **Training loop:** Standard PyTorch train/validation loop with early stopping.
- **Prediction:** Generates 30-step-ahead forecasts and inverse-transforms scaled outputs.

### `Models.py`
PyTorch `nn.Module` definitions for all three architectures:
- **FFNN** — fully connected layers with configurable depth and width
- **GRU** — stacked GRU with configurable layers and hidden units
- **Transformer** — encoder-only Transformer with tunable `d_model`, number of heads, and feedforward depth

### `utils.py`
Shared utilities: loading and splitting data chronologically, min-max scaling, computing MAPE/MOPE/MUPE, and generating Plotly prediction and error-analysis plots.

### `holidayCorrection.py`
Post-processing step applied after inference. Computes mean historical percentage change in demand for each day within a holiday window (from past years) and applies the correction to the raw model prediction. Reduces MAPE on holiday-affected test samples from ~20% to ~6.7%.

---

## Configuration Space

Each model is evaluated across combinations of:

| Dimension | Options |
|---|---|
| Input features | Sales only \| Sales + Crude oil price |
| Scaling | Min-max scaled \| Unscaled |
| Calendar encoding | None \| Ordinal (month 1–12, weekday 0–6) \| Cyclic (sin/cos) |

- **FFNN:** all 12 combinations (scaled + unscaled)
- **GRU / Transformer:** 6 combinations (scaled only — unscaled versions failed to converge)

---

## Workflow

Follow these steps in order:

### Step 1 — Prepare data
Place `DemandData.xlsx` and `CrudeOilPrice.xlsx` in `Data/`. Run `denoisingDataset.py` to generate the imputed and holiday-adjusted datasets.

### Step 2 — Single-objective hyperparameter tuning
Run the notebooks in `HyperParameterTuning/` (one per architecture). Each notebook runs an Optuna study minimising MAPE and saves the best hyperparameters per configuration.

### Step 3 — Evaluate over N runs
Run the notebooks in `Evaluation-MultipleRuns/`. Each notebook retrains the best hyperparameters from Step 2 over N random seeds and reports aggregated MAPE, MOPE, and MUPE.

### Step 4 — Multi-objective tuning (FFNN)
Inside `Multi Objective Optimization/`:
1. **`FFNN.ipynb`** — runs Optuna NSGAii to jointly minimise MAPE and MUPE, producing a Pareto front CSV per configuration.
2. **`best_hyperparameters_weighted.py`** — reads the Pareto CSVs and selects the top-5 ranked parameter sets per configuration using a weighted scoring scheme.
3. **`MultipleRuns.ipynb`** — evaluates the top-5 parameter sets over N runs each.
4. **`print_best_config_table.py`** — applies the same weighted scoring to the N-run results and prints the best configuration table.

### Step 5 — Training and inference
Run the notebooks in `Training and Inference/` to train the final selected hyperparameters, generate 30-day forecasts on the test set, apply holiday correction via `holidayCorrection.py`, and produce evaluation plots and summary tables.

---

## Loss Function — AMSE

The Asymmetric Mean Squared Error applies different penalty weights depending on the direction of error:

```
error  = (y_pred - y_true) / y_true

if error < 0  (under-prediction):  loss = α × MSE(y_pred, y_true)
if error >= 0 (over-prediction):   loss = β × MSE(y_pred, y_true)
```

α (`under_parameter`) and β (`over_parameter`) are tuned by Optuna. Under-prediction is operationally costlier (supply shortfall / contractual penalties), so multi-objective tuning favours solutions that reduce MUPE even at a small MAPE cost.

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| **MAPE** | Mean Absolute Percentage Error — overall forecast accuracy |
| **MOPE** | Mean Per-Sample Overprediction Percentage Error — average error magnitude on days where prediction > actual |
| **MUPE** | Mean Per-Sample Underprediction Percentage Error — average error magnitude on days where prediction < actual (reported as positive) |

---

## Key Results

### Best model per architecture (single-objective AMSE tuning, averaged over N runs)

| Model | Best Configuration | MAPE | MOPE | MUPE |
|---|---|---|---|---|
| **FFNN** | Sales + Crude, Scaled, Ordinal | **6.828%** | 7.716% | 5.085% |
| GRU | Sales + Crude, Scaled, Cyclic | 7.422% | 8.654% | 4.909% |
| Transformer | Sales + Crude, Scaled, None | 7.833% | 9.256% | 5.636% |

FFNN is the best overall architecture despite being the simplest.

### AMSE vs MSE ablation (FFNN, 12 configs)
- AMSE outperforms MSE on MAPE in **10/12** configurations
- AMSE outperforms MSE on MOPE in **11/12** configurations
- MUPE is higher under AMSE in most configs when using single-objective tuning (MAPE-only optimisation)

### Multi-objective tuning (FFNN, 12 configs)
- MUPE improves in **10/12** configurations, often by 1–2 percentage points
- MAPE and MOPE degrade by < 0.3% in most configurations
- **Recommended for deployment:** multi-objective tuning reduces supply-shortfall risk with negligible accuracy cost

### Holiday correction
- 164 of 335 test samples (~48.9%) have at least one holiday in their 30-day forecast horizon
- Average MAPE on holiday-affected samples: **19.99% → 6.72%** after post-processing correction

---

## Tech Stack

- **Python 3.x**
- **PyTorch** — model definition and training
- **Optuna** — hyperparameter tuning (TPE sampler, NSGAii for multi-objective)
- **Pandas / NumPy** — data processing
- **Plotly** — interactive prediction and error visualisations
- **Jupyter** — all tuning, evaluation, and inference workflows

---

## Splits

Chronological splits (no random shuffling — to prevent data leakage):

| Split | Period |
|---|---|
| Train | April 1, 2020 – May 31, 2023 |
| Validation | June 1, 2023 – December 31, 2023 |
| Test | January 1, 2024 – December 31, 2024 |
