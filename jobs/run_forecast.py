import json
import os
from pathlib import Path
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "load_forecasting_dataset_corrected.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "load_forecast_hgb.pkl"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "load_forecast_feature_columns.json"

FORECAST_SHORT_15MIN_PATH = BASE_DIR / "data" / "forecast_short_term_15min.csv"
FORECAST_SHORT_HOURLY_PATH = BASE_DIR / "data" / "forecast_short_term_hourly.csv"
FORECAST_MEDIUM_DAILY_PATH = BASE_DIR / "data" / "forecast_medium_term_daily.csv"
FORECAST_MEDIUM_MONTHLY_PATH = BASE_DIR / "data" / "forecast_medium_term_monthly.csv"
FORECAST_LONG_MONTHLY_PATH = BASE_DIR / "data" / "forecast_long_term_monthly.csv"
FORECAST_LONG_SCENARIOS_PATH = BASE_DIR / "data" / "forecast_long_term_scenarios.csv"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.json"

FREQUENCY = "15min"
SHORT_TERM_DAYS = 7
MEDIUM_TERM_DAYS = 180
LONG_TERM_MONTHS = 36
LAG_STEPS = [1, 4, 96, 192, 672]
ROLLING_WINDOWS = [4, 96, 672]
LONG_TERM_HISTORY_MONTHS = 36
LONG_TERM_SCENARIOS = {
    "low_demand": {
        "demand_multiplier": 0.96,
        "annual_growth": 0.006,
    },
    "baseline": {
        "demand_multiplier": 1.00,
        "annual_growth": 0.018,
    },
    "high_demand": {
        "demand_multiplier": 1.08,
        "annual_growth": 0.035,
    },
}

SHORT_TERM_EXTERNAL_FEATURE_PREFIXES = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Rainfall",
    "Solar Irradiance",
    "Electricity Price",
    "Public Event",
]

CALENDAR_FEATURE_COLUMNS = [
    "hour",
    "minute",
    "dayofweek",
    "day",
    "month",
    "quarter",
    "year",
    "dayofyear",
    "weekofyear",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_business_hours",
    "minute_of_day",
    "minute_of_week",
    "season_no",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]


def prepare_directories():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def model_artifacts_are_current(expected_feature_columns: list[str]) -> bool:
    if not (
        MODEL_PATH.exists()
        and FEATURE_COLUMNS_PATH.exists()
        and MODEL_PATH.stat().st_mtime >= DATA_PATH.stat().st_mtime
        and FEATURE_COLUMNS_PATH.stat().st_mtime >= DATA_PATH.stat().st_mtime
    ):
        return False

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        saved_feature_columns = json.load(f)

    return saved_feature_columns == expected_feature_columns


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "Timestamp" not in df.columns:
        raise ValueError("Dataset must contain Timestamp column")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET_COLUMN} column")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last")
    df = df.set_index("Timestamp")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    index = result.index

    result["hour"] = index.hour
    result["minute"] = index.minute
    result["dayofweek"] = index.dayofweek
    result["day"] = index.day
    result["month"] = index.month
    result["quarter"] = index.quarter
    result["year"] = index.year
    result["dayofyear"] = index.dayofyear
    result["weekofyear"] = index.isocalendar().week.astype(int)
    result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype(int)
    result["is_month_start"] = index.is_month_start.astype(int)
    result["is_month_end"] = index.is_month_end.astype(int)
    result["is_business_hours"] = result["hour"].between(9, 17).astype(int)
    result["minute_of_day"] = result["hour"] * 60 + result["minute"]
    result["minute_of_week"] = result["dayofweek"] * 24 * 60 + result["minute_of_day"]
    result["season_no"] = result["month"] % 12 // 3 + 1

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    return result


def numeric_external_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = set(df.select_dtypes(include=[np.number]).columns)
    selected_columns = []

    for prefix in SHORT_TERM_EXTERNAL_FEATURE_PREFIXES:
        matched_columns = [
            col for col in df.columns
            if col in numeric_columns and col.startswith(prefix)
        ]
        selected_columns.extend(matched_columns)

    return list(dict.fromkeys(selected_columns))


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    external_columns = numeric_external_columns(frame)
    calendar_columns = [
        col for col in CALENDAR_FEATURE_COLUMNS
        if col in frame.columns
    ]
    load_history_columns = [
        col for col in frame.columns
        if col.startswith("load_lag_") or col.startswith("load_rolling_")
    ]

    return external_columns + calendar_columns + load_history_columns


def add_load_lags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for lag in LAG_STEPS:
        result[f"load_lag_{lag}"] = result[TARGET_COLUMN].shift(lag)

    previous_load = result[TARGET_COLUMN].shift(1)

    for window in ROLLING_WINDOWS:
        result[f"load_rolling_mean_{window}"] = previous_load.rolling(window).mean()
        result[f"load_rolling_max_{window}"] = previous_load.rolling(window).max()
        result[f"load_rolling_min_{window}"] = previous_load.rolling(window).min()

    return result


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = add_calendar_features(df)
    frame = add_load_lags(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.dropna()


def calculate_metrics(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))

    denominator = np.where(actual == 0, np.nan, actual)
    mape = np.nanmean(np.abs((actual - forecast) / denominator)) * 100

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4)
    }


def train_short_term_model(frame: pd.DataFrame, feature_columns: list[str]):
    print("========== TRAIN SHORT/MEDIUM TERM MODEL ==========")

    validation_start = frame.index.max() - pd.Timedelta(days=90)
    train_df = frame[frame.index < validation_start].copy()
    test_df = frame[frame.index >= validation_start].copy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42
    )

    model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    validation_forecast = model.predict(test_df[feature_columns])
    metrics = calculate_metrics(test_df[TARGET_COLUMN].values, validation_forecast)

    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=42
    )
    final_model.fit(frame[feature_columns], frame[TARGET_COLUMN])

    joblib.dump(final_model, MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    print("Validation start:", validation_start)
    print("Validation metrics:", metrics)
    print("Model saved:", MODEL_PATH.name)

    return final_model, feature_columns, metrics, validation_start


def load_short_term_model():
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return joblib.load(MODEL_PATH), feature_columns


def build_future_external_features(df: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    external_columns = numeric_external_columns(df)

    history = df[external_columns].copy()
    history["profile_month"] = history.index.month
    history["profile_dayofweek"] = history.index.dayofweek
    history["profile_hour"] = history.index.hour
    history["profile_minute"] = history.index.minute

    keys = ["profile_month", "profile_dayofweek", "profile_hour", "profile_minute"]
    profile = history.groupby(keys)[external_columns].mean().reset_index()

    future = pd.DataFrame(index=future_index)
    future["profile_month"] = future.index.month
    future["profile_dayofweek"] = future.index.dayofweek
    future["profile_hour"] = future.index.hour
    future["profile_minute"] = future.index.minute

    future = (
        future
        .reset_index(names="Timestamp")
        .merge(profile, on=keys, how="left")
        .set_index("Timestamp")
        .sort_index()
    )

    means = df[external_columns].mean(numeric_only=True)
    for col in external_columns:
        future[col] = future[col].fillna(means[col])

    future = future.drop(columns=keys)
    future = add_calendar_features(future)

    return future


def build_expected_load_series(df: pd.DataFrame, extended_index: pd.DatetimeIndex) -> pd.Series:
    history = df[[TARGET_COLUMN]].copy()
    history["profile_month"] = history.index.month
    history["profile_dayofweek"] = history.index.dayofweek
    history["profile_hour"] = history.index.hour
    history["profile_minute"] = history.index.minute

    keys = ["profile_month", "profile_dayofweek", "profile_hour", "profile_minute"]
    profile = history.groupby(keys)[TARGET_COLUMN].mean().reset_index()

    expected = pd.DataFrame(index=extended_index)
    expected["profile_month"] = expected.index.month
    expected["profile_dayofweek"] = expected.index.dayofweek
    expected["profile_hour"] = expected.index.hour
    expected["profile_minute"] = expected.index.minute

    expected = (
        expected
        .reset_index(names="Timestamp")
        .merge(profile, on=keys, how="left")
        .set_index("Timestamp")
        .sort_index()
    )

    expected_load = expected[TARGET_COLUMN].fillna(df[TARGET_COLUMN].mean())
    actual_overlap = df[TARGET_COLUMN].reindex(extended_index)
    expected_load.loc[actual_overlap.notna()] = actual_overlap.dropna()

    return expected_load


def build_future_feature_frame(
    df: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    feature_columns: list[str],
) -> pd.DataFrame:
    future_features = build_future_external_features(df, future_index)

    max_history = max(max(LAG_STEPS), max(ROLLING_WINDOWS))
    extended_index = pd.date_range(
        start=future_index.min() - pd.Timedelta(minutes=15 * max_history),
        end=future_index.max(),
        freq=FREQUENCY,
    )
    expected_load = build_expected_load_series(df, extended_index)

    lag_frame = pd.DataFrame(index=extended_index)
    lag_frame[TARGET_COLUMN] = expected_load

    for lag in LAG_STEPS:
        lag_frame[f"load_lag_{lag}"] = lag_frame[TARGET_COLUMN].shift(lag)

    previous_load = lag_frame[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        lag_frame[f"load_rolling_mean_{window}"] = previous_load.rolling(window).mean()
        lag_frame[f"load_rolling_max_{window}"] = previous_load.rolling(window).max()
        lag_frame[f"load_rolling_min_{window}"] = previous_load.rolling(window).min()

    future_frame = future_features.join(
        lag_frame.drop(columns=[TARGET_COLUMN]).reindex(future_index),
        how="left",
    )

    return future_frame.reindex(columns=feature_columns)


def forecast_future_15min(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    periods: int,
) -> pd.DataFrame:
    start = df.index.max() + pd.Timedelta(minutes=15)
    future_index = pd.date_range(start=start, periods=periods, freq=FREQUENCY)
    future_frame = build_future_feature_frame(df, future_index, feature_columns)
    forecast_values = model.predict(future_frame)

    result = pd.DataFrame({
        "Timestamp": future_index,
        "forecast_load": forecast_values,
    })
    result["horizon"] = "future"
    result["actual_load"] = np.nan
    result["error"] = np.nan
    result["error_percent"] = np.nan

    return result[
        ["Timestamp", "actual_load", "forecast_load", "error", "error_percent", "horizon"]
    ]


def write_short_term_outputs(forecast_15min: pd.DataFrame) -> pd.DataFrame:
    short = forecast_15min.head(SHORT_TERM_DAYS * 96).copy()
    short.to_csv(FORECAST_SHORT_15MIN_PATH, index=False)

    hourly = short.copy()
    hourly["Timestamp"] = pd.to_datetime(hourly["Timestamp"])
    hourly["hour_bucket"] = hourly["Timestamp"].dt.floor("h")
    hourly = (
        hourly
        .groupby("hour_bucket", as_index=False)
        .agg(forecast_load=("forecast_load", "mean"))
        .rename(columns={"hour_bucket": "Timestamp"})
    )

    hourly["actual_load"] = np.nan
    hourly["error"] = np.nan
    hourly["error_percent"] = np.nan
    hourly = hourly[["Timestamp", "actual_load", "forecast_load", "error", "error_percent"]]
    hourly.to_csv(FORECAST_SHORT_HOURLY_PATH, index=False)

    return short


def write_medium_term_outputs(forecast_15min: pd.DataFrame):
    df = forecast_15min.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    daily = df.copy()
    daily["day_bucket"] = daily["Timestamp"].dt.floor("D")
    daily = (
        daily
        .groupby("day_bucket", as_index=False)
        .agg(
            forecast_peak_load=("forecast_load", "max"),
            forecast_avg_load=("forecast_load", "mean"),
        )
        .rename(columns={"day_bucket": "Timestamp"})
    )
    daily["actual_peak_load"] = np.nan
    daily["actual_avg_load"] = np.nan
    daily["error_percent"] = np.nan
    daily = daily[
        [
            "Timestamp",
            "actual_peak_load",
            "forecast_peak_load",
            "actual_avg_load",
            "forecast_avg_load",
            "error_percent",
        ]
    ]
    daily.to_csv(FORECAST_MEDIUM_DAILY_PATH, index=False)

    monthly = df.copy()
    monthly["month_bucket"] = monthly["Timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        monthly
        .groupby("month_bucket", as_index=False)
        .agg(
            forecast_peak_load=("forecast_load", "max"),
            forecast_avg_load=("forecast_load", "mean"),
        )
        .rename(columns={"month_bucket": "Timestamp"})
    )
    monthly["actual_peak_load"] = np.nan
    monthly["actual_avg_load"] = np.nan
    monthly["error_percent"] = np.nan
    monthly = monthly[
        [
            "Timestamp",
            "actual_peak_load",
            "forecast_peak_load",
            "actual_avg_load",
            "forecast_avg_load",
            "error_percent",
        ]
    ]
    monthly.to_csv(FORECAST_MEDIUM_MONTHLY_PATH, index=False)


def build_long_term_scenarios(df: pd.DataFrame):
    monthly = (
        df[TARGET_COLUMN]
        .resample("MS")
        .agg(actual_avg_load="mean", actual_peak_load="max")
        .dropna()
        .reset_index()
    )

    future_months = pd.date_range(
        start=(df.index.max() + pd.offsets.MonthBegin(1)).normalize(),
        periods=LONG_TERM_MONTHS,
        freq="MS",
    )

    monthly["month"] = monthly["Timestamp"].dt.month
    history = monthly.tail(LONG_TERM_HISTORY_MONTHS).copy()

    avg_overall = history["actual_avg_load"].mean()
    peak_overall = history["actual_peak_load"].mean()
    avg_seasonal_factor = (
        history.groupby("month")["actual_avg_load"].mean() / avg_overall
    ).to_dict()
    peak_seasonal_factor = (
        history.groupby("month")["actual_peak_load"].mean() / peak_overall
    ).to_dict()

    avg_base_level = (
        history["actual_avg_load"]
        / history["month"].map(avg_seasonal_factor)
    ).mean()
    peak_base_level = (
        history["actual_peak_load"]
        / history["month"].map(peak_seasonal_factor)
    ).mean()

    scenario_rows = []
    for scenario, config in LONG_TERM_SCENARIOS.items():
        multiplier = config["demand_multiplier"]
        annual_growth = config["annual_growth"]

        for month_index, timestamp in enumerate(future_months, start=1):
            years_ahead = month_index / 12
            growth_factor = (1 + annual_growth) ** years_ahead
            month = timestamp.month

            avg_load = (
                avg_base_level
                * avg_seasonal_factor.get(month, 1.0)
                * growth_factor
                * multiplier
            )
            peak_load = (
                peak_base_level
                * peak_seasonal_factor.get(month, 1.0)
                * growth_factor
                * multiplier
            )

            scenario_rows.append({
                "Timestamp": timestamp,
                "scenario": scenario,
                "forecast_avg_load": max(float(avg_load), 0.0),
                "forecast_peak_load": max(float(peak_load), 0.0),
            })

    scenarios_df = pd.DataFrame(scenario_rows)
    scenarios_df.to_csv(FORECAST_LONG_SCENARIOS_PATH, index=False)

    baseline = scenarios_df[scenarios_df["scenario"] == "baseline"].copy()
    baseline = baseline.drop(columns=["scenario"])
    baseline.to_csv(FORECAST_LONG_MONTHLY_PATH, index=False)


def run_forecast():
    print("========== START LOAD FORECAST JOB ==========")
    print("Start time:", datetime.now())

    prepare_directories()

    df = load_dataset()
    frame = build_training_frame(df)
    expected_feature_columns = model_feature_columns(frame)

    print("Dataset shape:", df.shape)
    print("Training frame shape:", frame.shape)
    print("Selected feature count:", len(expected_feature_columns))
    print("Min time:", df.index.min())
    print("Max time:", df.index.max())

    if model_artifacts_are_current(expected_feature_columns):
        print("Current model artifact found. Loading model...")
        model, feature_columns = load_short_term_model()
        previous_metrics = {}
        validation_start = None

        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                previous_metrics = json.load(f)

            validation_start = previous_metrics.get("validation_start")
            metrics = previous_metrics.get("metrics", {})
        else:
            metrics = {}
    else:
        model, feature_columns, metrics, validation_start = train_short_term_model(
            frame,
            expected_feature_columns,
        )

    forecast_15min = forecast_future_15min(
        model=model,
        df=df,
        feature_columns=feature_columns,
        periods=MEDIUM_TERM_DAYS * 96,
    )

    short = write_short_term_outputs(forecast_15min)
    write_medium_term_outputs(forecast_15min)
    build_long_term_scenarios(df)

    metrics_payload = {
        "mode": "forecast_from_dataset_end",
        "model_name": "HistGradientBoostingRegressor",
        "data_start": str(df.index.min()),
        "data_end": str(df.index.max()),
        "forecast_start": str(forecast_15min["Timestamp"].min()),
        "short_term_end": str(short["Timestamp"].max()),
        "medium_term_end": str(forecast_15min["Timestamp"].max()),
        "long_term_months": LONG_TERM_MONTHS,
        "validation_start": str(validation_start) if validation_start else None,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "generated_at": datetime.now().isoformat()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print("Metrics:", metrics_payload)
    print("End time:", datetime.now())
    print("========== END LOAD FORECAST JOB ==========")


if __name__ == "__main__":
    run_forecast()
