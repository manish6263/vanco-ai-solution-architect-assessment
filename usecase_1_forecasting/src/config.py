"""Configuration values for the Store Sales forecasting pipeline."""

from pathlib import Path


USECASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = USECASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = USECASE_DIR / "submissions"
REPORTS_DIR = USECASE_DIR / "reports"
SCREENSHOTS_DIR = USECASE_DIR / "screenshots"

TARGET = "sales"
ID_COLUMN = "id"
DATE_COLUMN = "date"

FORECAST_HORIZON_DAYS = 16
RANDOM_SEED = 42

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
    "sample_submission": "sample_submission.csv",
}

