"""Build joined Store Sales datasets from the raw Kaggle tables."""

from __future__ import annotations

import pandas as pd

from .config import DATE_COLUMN, ID_COLUMN, PROCESSED_DATA_DIR, TARGET
from .data_loader import load_all


def prepare_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Create a complete daily oil-price table with forward/backward filling."""
    oil = oil.sort_values(DATE_COLUMN).copy()
    full_dates = pd.DataFrame(
        {DATE_COLUMN: pd.date_range(oil[DATE_COLUMN].min(), oil[DATE_COLUMN].max(), freq="D")}
    )
    oil = full_dates.merge(oil, on=DATE_COLUMN, how="left")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    return oil


def _aggregate_holiday_flags(holidays: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Aggregate holiday rows to one row per date and locale name."""
    subset = holidays.loc[
        (holidays["locale"] == locale) & (~holidays["transferred"].astype(bool))
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=[DATE_COLUMN, "locale_name"])

    prefix = locale.lower()
    grouped = (
        subset.assign(
            **{
                f"is_{prefix}_holiday": 1,
                f"{prefix}_holiday_type": subset["type"].astype(str),
                f"{prefix}_holiday_description": subset["description"].astype(str),
            }
        )
        .groupby([DATE_COLUMN, "locale_name"], as_index=False)
        .agg(
            {
                f"is_{prefix}_holiday": "max",
                f"{prefix}_holiday_type": lambda values: "|".join(sorted(set(values))),
                f"{prefix}_holiday_description": lambda values: "|".join(
                    sorted(set(values))
                ),
            }
        )
    )
    return grouped


def add_holiday_features(frame: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Add national, regional, and local holiday flags without duplicating rows."""
    result = frame.copy()
    active_holidays = holidays.loc[~holidays["transferred"].astype(bool)].copy()

    national = active_holidays.loc[active_holidays["locale"] == "National"].copy()
    if not national.empty:
        national_features = (
            national.assign(
                is_national_holiday=1,
                national_holiday_type=national["type"].astype(str),
                national_holiday_description=national["description"].astype(str),
            )
            .groupby(DATE_COLUMN, as_index=False)
            .agg(
                {
                    "is_national_holiday": "max",
                    "national_holiday_type": lambda values: "|".join(
                        sorted(set(values))
                    ),
                    "national_holiday_description": lambda values: "|".join(
                        sorted(set(values))
                    ),
                }
            )
        )
        result = result.merge(national_features, on=DATE_COLUMN, how="left")

    regional = _aggregate_holiday_flags(active_holidays, "Regional").rename(
        columns={"locale_name": "state"}
    )
    if not regional.empty:
        result = result.merge(regional, on=[DATE_COLUMN, "state"], how="left")

    local = _aggregate_holiday_flags(active_holidays, "Local").rename(
        columns={"locale_name": "city"}
    )
    if not local.empty:
        result = result.merge(local, on=[DATE_COLUMN, "city"], how="left")

    holiday_flag_columns = [
        col for col in result.columns if col.startswith("is_") and col != "is_train"
    ]
    for col in holiday_flag_columns:
        result[col] = result[col].fillna(0).astype("int8")

    text_columns = [col for col in result.columns if col.endswith("_description")]
    text_columns += [col for col in result.columns if col.endswith("_type")]
    for col in text_columns:
        result[col] = result[col].fillna("none")

    return result


def build_modeling_frame(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one joined train/test frame with source markers and raw external features."""
    train = datasets["train"].copy()
    test = datasets["test"].copy()

    train["is_train"] = True
    test["is_train"] = False
    test[TARGET] = float("nan")

    base_columns = [ID_COLUMN, DATE_COLUMN, "store_nbr", "family", "onpromotion", TARGET, "is_train"]
    frame = pd.concat([train[base_columns], test[base_columns]], ignore_index=True)
    frame = frame.sort_values(["store_nbr", "family", DATE_COLUMN]).reset_index(drop=True)

    frame = frame.merge(datasets["stores"], on="store_nbr", how="left", validate="many_to_one")
    frame = frame.merge(prepare_oil(datasets["oil"]), on=DATE_COLUMN, how="left")
    frame = frame.merge(
        datasets["transactions"],
        on=[DATE_COLUMN, "store_nbr"],
        how="left",
        validate="many_to_one",
    )
    frame = add_holiday_features(frame, datasets["holidays"])

    return frame


def write_modeling_frame(frame: pd.DataFrame) -> None:
    """Persist the joined base frame for faster downstream experimentation."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(PROCESSED_DATA_DIR / "modeling_base.parquet", index=False)


def main() -> None:
    datasets = load_all()
    frame = build_modeling_frame(datasets)
    write_modeling_frame(frame)
    print(f"Wrote {len(frame):,} rows to {PROCESSED_DATA_DIR / 'modeling_base.parquet'}")


if __name__ == "__main__":
    main()
