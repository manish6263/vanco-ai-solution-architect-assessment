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

DATE_COLUMNS = {
    "train": ["date"],
    "test": ["date"],
    "oil": ["date"],
    "holidays": ["date"],
    "transactions": ["date"],
}

EXPECTED_COLUMNS = {
    "train": {"id", "date", "store_nbr", "family", "sales", "onpromotion"},
    "test": {"id", "date", "store_nbr", "family", "onpromotion"},
    "stores": {"store_nbr", "city", "state", "type", "cluster"},
    "oil": {"date", "dcoilwtico"},
    "holidays": {
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    },
    "transactions": {"date", "store_nbr", "transactions"},
    "sample_submission": {"id", "sales"},
}
