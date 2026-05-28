# Use Case 1: Grocery Sales Forecasting With External Events

## Objective

Build a forecasting model for store/product-family sales using the Kaggle Store Sales - Time Series Forecasting dataset and external event features.

## Planned Architecture

```text
Kaggle CSV tables
    -> data validation and joins
    -> calendar/event/oil/promotion/lag feature generation
    -> time-aware backtesting
    -> baseline model
    -> gradient boosting model
    -> error analysis and feature importance
    -> Kaggle submission
```

## Data

Expected Kaggle files:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`
- `sample_submission.csv`

Raw data should be placed under `data/raw/`. Large Kaggle files should not be committed unless explicitly allowed.

### Download Option A: Kaggle Website

1. Open the competition page:
   `https://www.kaggle.com/competitions/store-sales-time-series-forecasting`
2. Accept the competition rules if Kaggle asks.
3. Download the dataset files.
4. Extract/copy the CSV files into:

```text
usecase_1_forecasting/data/raw/
```

### Download Option B: Kaggle CLI

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure Kaggle credentials by placing `kaggle.json` in the standard Kaggle config location:

```text
Windows: C:\Users\<your-user>\.kaggle\kaggle.json
Linux/macOS: ~/.kaggle/kaggle.json
```

Then run:

```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
```

Extract the downloaded zip so `data/raw/` contains the expected CSV files.

Validate the local data files:

```bash
python -m src.validate_data
```

After validation passes, inspect the raw data and joined frame:

```bash
python -m src.summarize_data
```

Create the first joined modeling table:

```bash
python -m src.dataset
```

## Repository Layout

```text
usecase_1_forecasting/
|-- README.md
|-- requirements.txt
|-- notebooks/
|   `-- 01_store_sales_forecasting.ipynb
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- dataset.py
|   |-- features.py
|   |-- validation.py
|   |-- metrics.py
|   |-- models.py
|   |-- train.py
|   |-- predict.py
|   |-- summarize_data.py
|   |-- smoke_test.py
|   |-- validate_data.py
|   `-- analysis.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- submissions/
|-- screenshots/
`-- reports/
```

## Setup

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The implementation scripts will be runnable as the pipeline is completed:

```bash
python -m src.smoke_test
python -m src.validate_data
python -m src.summarize_data
python -m src.dataset
python -m src.train
python -m src.predict
```

## Validation Strategy

The solution will use time-aware validation/backtesting. Random train/validation splits are avoided because they leak future information into training.

Planned validation:

- Primary holdout window: final 16 days of the training period, matching the Kaggle forecast horizon
- Baselines evaluated on the same holdout window
- One or more rolling backtest windows for model robustness checks
- Model comparison using RMSLE and business-oriented error breakdowns

## Feature Strategy

Planned features:

- Store, city, state, type, cluster
- Product family
- Calendar features
- Holiday/event flags
- Oil price lags and rolling statistics
- Promotion features
- Sales lags and rolling statistics by store-family
- Seasonality features

Important leakage rule: transaction values are only available historically. They may be used directly for validation rows where the date is known in `transactions.csv`, but final test-time features must rely on lagged or aggregated historical transaction information rather than unknown future transactions.

## Modeling Plan

- Baselines: store-family mean, last value, 7-day seasonal naive, 28-day rolling mean
- Main model: LightGBM, CatBoost, or XGBoost
- Optional: ensemble with baseline corrections

## Deliverables

- [ ] Notebook or scripts for training and inference
- [ ] Kaggle submission file
- [ ] Leaderboard screenshot
- [ ] Pipeline diagram
- [ ] Feature importance/explainability
- [ ] Error analysis
- [ ] Limitations and improvement plan
