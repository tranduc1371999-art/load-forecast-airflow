import pandas as pd


TARGET_COLUMN = "Load Demand (kW)"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["year"] = df.index.year
    df["season_no"] = df["month"] % 12 // 3 + 1
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_start"] = (df["dayofmonth"] == 1).astype(int)
    df["is_month_end"] = (df["dayofmonth"] == df.index.days_in_month).astype(int)

    df["is_working_day"] = df["dayofweek"].isin([0, 1, 2, 3, 4]).astype(int)
    df["is_business_hours"] = df["hour"].between(9, 17).astype(int)
    df["is_peak_hour"] = df["hour"].isin([8, 12, 18]).astype(int)

    df["minute_of_day"] = df["hour"] * 60 + df["minute"]
    df["minute_of_week"] = df["dayofweek"] * 24 * 60 + df["minute_of_day"]

    return df


def load_and_prepare_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "Timestamp" not in df.columns:
        raise ValueError("Dataset must contain Timestamp column")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET_COLUMN} column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df.set_index("Timestamp")

    df = create_features(df)

    if "Season" in df.columns:
        df = pd.get_dummies(df, columns=["Season"], prefix="season", dtype=int)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame):
    return [col for col in df.columns if col != TARGET_COLUMN]