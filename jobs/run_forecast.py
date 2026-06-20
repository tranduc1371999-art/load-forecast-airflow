import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from model import LSTMModel
from preprocessing import load_and_prepare_dataset, get_feature_columns, TARGET_COLUMN


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "load_forecasting_dataset_corrected.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "lstm_model.pth"
SCALER_X_PATH = ARTIFACT_DIR / "scaler_X.pkl"
SCALER_Y_PATH = ARTIFACT_DIR / "scaler_y.pkl"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"

FORECAST_15MIN_PATH = BASE_DIR / "data" / "forecast_15min.csv"
FORECAST_CURRENT_PATH = BASE_DIR / "data" / "forecast_current_15min.csv"
FORECAST_HISTORY_PATH = BASE_DIR / "data" / "forecast_history_15min.csv"

FORECAST_HOURLY_PATH = BASE_DIR / "data" / "forecast_hourly.csv"
FORECAST_DAILY_PATH = BASE_DIR / "data" / "forecast_daily.csv"
FORECAST_MONTHLY_PATH = BASE_DIR / "data" / "forecast_monthly.csv"
METRICS_PATH = BASE_DIR / "data" / "model_metrics.json"

SIMULATION_LAG_YEARS = 2
REALTIME_BUCKET_MINUTES = 15


def prepare_directories():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)


def artifacts_exist():
    return (
        MODEL_PATH.exists()
        and SCALER_X_PATH.exists()
        and SCALER_Y_PATH.exists()
        and FEATURE_COLUMNS_PATH.exists()
    )


def get_simulated_now():
    real_now = pd.Timestamp.now().tz_localize(None)
    simulated_now = real_now - pd.DateOffset(years=SIMULATION_LAG_YEARS)
    simulated_now = simulated_now.floor(f"{REALTIME_BUCKET_MINUTES}min")
    return real_now, simulated_now


def build_tensors(x_scaled, y_scaled):
    x_seq = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
    x_tensor = torch.from_numpy(x_seq).float()
    y_tensor = torch.from_numpy(y_scaled).float()
    return x_tensor, y_tensor


def train_model(df: pd.DataFrame, epochs: int = 10, batch_size: int = 256):
    print("========== TRAIN LSTM MODEL ==========")

    feature_columns = get_feature_columns(df)

    train_df = df[df.index < "2025-01-01"].copy()
    test_df = df[df.index >= "2025-01-01"].copy()

    if train_df.empty:
        raise ValueError("Training dataset is empty")

    if test_df.empty:
        raise ValueError("Testing dataset is empty")

    x_train = train_df[feature_columns]
    y_train = train_df[[TARGET_COLUMN]]

    x_test = test_df[feature_columns]
    y_test = test_df[[TARGET_COLUMN]]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    x_test_scaled = scaler_x.transform(x_test)
    y_test_scaled = scaler_y.transform(y_test)

    x_train_tensor, y_train_tensor = build_tensors(x_train_scaled, y_train_scaled)
    x_test_tensor, y_test_tensor = build_tensors(x_test_scaled, y_test_scaled)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(x_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = LSTMModel(
        input_size=x_train_tensor.shape[2],
        hidden_size=50,
        num_layers=1
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(test_loader.dataset)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler_x, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    print("Model saved:", MODEL_PATH)
    print("Scaler X saved:", SCALER_X_PATH)
    print("Scaler y saved:", SCALER_Y_PATH)
    print("Feature columns saved:", FEATURE_COLUMNS_PATH)

    return model, scaler_x, scaler_y, feature_columns


def load_model_and_artifacts(input_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(
        input_size=input_size,
        hidden_size=50,
        num_layers=1
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler_x, scaler_y, feature_columns


def predict_dataframe(model, scaler_x, scaler_y, df: pd.DataFrame, feature_columns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = df[feature_columns]
    x_scaled = scaler_x.transform(x)

    fake_y = np.zeros((len(x_scaled), 1))
    x_tensor, _ = build_tensors(x_scaled, fake_y)

    x_tensor = x_tensor.to(device)

    predictions = []

    model.eval()
    with torch.no_grad():
        batch_size = 512

        for start in range(0, len(x_tensor), batch_size):
            batch = x_tensor[start:start + batch_size]
            output = model(batch)
            predictions.append(output.cpu().numpy())

    predictions_scaled = np.vstack(predictions)
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

    return predictions


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


def build_hourly_output(result_target: pd.DataFrame):
    df = result_target.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    df["hour_bucket"] = df["Timestamp"].dt.floor("h")

    hourly = (
        df
        .groupby("hour_bucket", as_index=False)
        .agg(
            actual_load=("actual_load", "mean"),
            forecast_load=("forecast_load", "mean"),
            error=("error", "mean"),
            error_percent=("error_percent", "mean")
        )
        .rename(columns={"hour_bucket": "Timestamp"})
    )

    hourly.to_csv(FORECAST_HOURLY_PATH, index=False)

    print("Saved hourly forecast:", FORECAST_HOURLY_PATH)


def build_daily_monthly_output(history_df: pd.DataFrame):
    df = history_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")

    if "error" not in df.columns:
        df["error"] = df["actual_load"] - df["forecast_load"]

    if "error_percent" not in df.columns:
        df["error_percent"] = df["error"].abs() / df["actual_load"] * 100

    df["day_bucket"] = df["Timestamp"].dt.floor("d")

    daily = (
        df
        .groupby("day_bucket", as_index=False)
        .agg(
            actual_peak_load=("actual_load", "max"),
            forecast_peak_load=("forecast_load", "max"),
            actual_avg_load=("actual_load", "mean"),
            forecast_avg_load=("forecast_load", "mean"),
            error_percent=("error_percent", "mean")
        )
        .rename(columns={"day_bucket": "Timestamp"})
    )

    daily.to_csv(FORECAST_DAILY_PATH, index=False)

    df["month_bucket"] = df["Timestamp"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df
        .groupby("month_bucket", as_index=False)
        .agg(
            actual_peak_load=("actual_load", "max"),
            forecast_peak_load=("forecast_load", "max"),
            actual_avg_load=("actual_load", "mean"),
            forecast_avg_load=("forecast_load", "mean"),
            error_percent=("error_percent", "mean")
        )
        .rename(columns={"month_bucket": "Timestamp"})
    )

    monthly.to_csv(FORECAST_MONTHLY_PATH, index=False)

    print("Saved daily forecast:", FORECAST_DAILY_PATH)
    print("Saved monthly forecast:", FORECAST_MONTHLY_PATH)


def run_forecast():
    print("========== START LOAD FORECAST REALTIME -2 YEARS SIMULATION JOB ==========")
    print("Start time:", datetime.now())

    prepare_directories()

    df = load_and_prepare_dataset(str(DATA_PATH))

    print("Dataset shape:", df.shape)
    print("Min time:", df.index.min())
    print("Max time:", df.index.max())

    if not artifacts_exist():
        print("Model artifact not found or incomplete. Training model first...")

        model, scaler_x, scaler_y, feature_columns = train_model(
            df=df,
            epochs=10,
            batch_size=256
        )
    else:
        print("Model artifact found. Loading model...")

        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            feature_columns = json.load(f)

        model, scaler_x, scaler_y, feature_columns = load_model_and_artifacts(
            input_size=len(feature_columns)
        )

    real_now, simulated_now = get_simulated_now()

    current_date = simulated_now.date()
    target_date = current_date + timedelta(days=1)

    print("Real now:", real_now)
    print("Simulated now:", simulated_now)
    print("Current simulation date:", current_date)
    print("Forecast target date:", target_date)

    target_df = df[df.index.date == target_date].copy()

    if target_df.empty:
        raise ValueError(f"No data found for forecast target date: {target_date}")

    forecast_values = predict_dataframe(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        df=target_df,
        feature_columns=feature_columns
    )

    actual_values = target_df[TARGET_COLUMN].values

    result_target = pd.DataFrame({
        "Timestamp": target_df.index,
        "actual_load": actual_values,
        "forecast_load": forecast_values
    })

    result_target["error"] = result_target["actual_load"] - result_target["forecast_load"]
    result_target["error_percent"] = (
        result_target["error"].abs() / result_target["actual_load"] * 100
    )

    result_target.to_csv(FORECAST_15MIN_PATH, index=False)
    result_target.to_csv(FORECAST_CURRENT_PATH, index=False)

    if FORECAST_HISTORY_PATH.exists():
        history_df = pd.read_csv(FORECAST_HISTORY_PATH)
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"])

        combined_df = pd.concat([history_df, result_target], ignore_index=True)
        combined_df["Timestamp"] = pd.to_datetime(combined_df["Timestamp"])

        combined_df = (
            combined_df
            .sort_values("Timestamp")
            .drop_duplicates(subset=["Timestamp"], keep="last")
        )
    else:
        combined_df = result_target.copy()

    combined_df.to_csv(FORECAST_HISTORY_PATH, index=False)

    build_hourly_output(result_target)
    build_daily_monthly_output(combined_df)

    metrics = calculate_metrics(
        actual=result_target["actual_load"].values,
        forecast=result_target["forecast_load"].values
    )

    metrics_payload = {
        "mode": "real_time_minus_2_years_simulation",
        "model_name": "PyTorch LSTM",
        "real_now": str(real_now),
        "simulated_now": str(simulated_now),
        "current_date": str(current_date),
        "forecast_target_date": str(target_date),
        "train_period": "2020-01-01 to 2024-12-31",
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "generated_at": datetime.now().isoformat()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print("Metrics:", metrics_payload)

    print("End time:", datetime.now())
    print("========== END LOAD FORECAST REALTIME -2 YEARS SIMULATION JOB ==========")


if __name__ == "__main__":
    run_forecast()
