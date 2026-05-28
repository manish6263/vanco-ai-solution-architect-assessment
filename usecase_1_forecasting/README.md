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

