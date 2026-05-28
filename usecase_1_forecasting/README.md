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

## Repository Layout

```text
usecase_1_forecasting/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_store_sales_forecasting.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── validation.py
│   ├── metrics.py
│   ├── models.py
│   ├── train.py
│   ├── predict.py
│   └── analysis.py
├── data/
│   ├── raw/
│   └── processed/
├── submissions/
├── screenshots/
└── reports/
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
python -m src.train
python -m src.predict
```

## Validation Strategy

The solution will use time-aware validation/backtesting. Random train/validation splits are avoided because they leak future information into training.

Planned validation:

- Seasonal naive baseline on a recent holdout window
- One or more rolling backtest windows
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

## Modeling Plan

- Baseline: seasonal naive or rolling mean
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
