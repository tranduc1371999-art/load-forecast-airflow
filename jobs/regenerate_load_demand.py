from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "load_forecasting_dataset_corrected.csv"


def column_by_prefix(df: pd.DataFrame, prefix: str) -> str:
    matches = [col for col in df.columns if col.startswith(prefix)]
    if not matches:
        raise ValueError(f"Missing column starting with: {prefix}")
    return matches[0]


def build_hour_profile(timestamp: pd.Series) -> np.ndarray:
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute

    # Sri Lanka style synthetic profile:
    # low overnight, morning ramp, working-hour plateau, strongest evening peak.
    profile_minutes = np.array([
        0,
        180,
        330,
        450,
        540,
        720,
        960,
        1110,
        1230,
        1350,
        1439,
    ])
    profile_load = np.array([
        -210,
        -250,
        -180,
        40,
        150,
        190,
        160,
        260,
        430,
        250,
        -80,
    ])

    return np.interp(minute_of_day, profile_minutes, profile_load)


def regenerate_load_demand(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    timestamp = pd.to_datetime(df["Timestamp"])
    temp_col = column_by_prefix(df, "Temperature")
    humidity_col = column_by_prefix(df, "Humidity")
    rainfall_col = column_by_prefix(df, "Rainfall")
    solar_col = column_by_prefix(df, "Solar Irradiance")

    rng = np.random.default_rng(seed)

    hour_profile = build_hour_profile(timestamp)

    dayofweek = timestamp.dt.dayofweek
    weekend_effect = np.where(dayofweek.isin([5, 6]), -90, 35)

    month = timestamp.dt.month
    seasonal_effect = np.select(
        [
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8, 9]),
            month.isin([12, 1, 2]),
        ],
        [
            90,
            45,
            -35,
        ],
        default=15,
    )

    temperature = pd.to_numeric(df[temp_col], errors="coerce")
    humidity = pd.to_numeric(df[humidity_col], errors="coerce")
    rainfall = pd.to_numeric(df[rainfall_col], errors="coerce")
    solar = pd.to_numeric(df[solar_col], errors="coerce")
    event = pd.to_numeric(df.get("Public Event", 0), errors="coerce").fillna(0)

    cooling_effect = np.maximum(temperature - 27.0, 0) * 32
    humidity_effect = np.maximum(humidity - 78.0, 0) * 1.8
    rainfall_effect = np.minimum(rainfall, 20) * 2.5
    solar_effect = np.clip(solar - 550, 0, None) * 0.08
    event_effect = event * 100

    weekly_wave = 25 * np.sin(2 * np.pi * dayofweek / 7)
    annual_wave = 35 * np.sin(2 * np.pi * timestamp.dt.dayofyear / 365.25)
    noise = rng.normal(0, 35, len(df))

    load = (
        1250
        + hour_profile
        + weekend_effect
        + seasonal_effect
        + cooling_effect
        + humidity_effect
        + rainfall_effect
        + solar_effect
        + event_effect
        + weekly_wave
        + annual_wave
        + noise
    )

    return pd.Series(np.clip(load, 750, 2100), index=df.index).round(3)


def main():
    df = pd.read_csv(DATA_PATH)
    df[TARGET_COLUMN] = regenerate_load_demand(df)
    df.to_csv(DATA_PATH, index=False)

    print(f"Updated target column in {DATA_PATH.name}")
    print(df[TARGET_COLUMN].describe().round(2).to_string())


if __name__ == "__main__":
    main()
