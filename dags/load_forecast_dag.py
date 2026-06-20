from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


AIRFLOW_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = AIRFLOW_DIR / "jobs"

sys.path.append(str(JOBS_DIR))

from run_forecast import run_forecast


default_args = {
    "owner": "ductd",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="load_forecast_realtime_minus_2_years_dag",
    default_args=default_args,
    description="Realtime simulation: current time minus 2 years, LSTM day-ahead forecast",
    start_date=datetime(2025, 1, 1),
    schedule="*/15 * * * *",
    catchup=False,
    tags=["load-forecast", "lstm", "realtime-simulation"],
) as dag:

    run_forecast_task = PythonOperator(
        task_id="run_realtime_minus_2_years_forecast",
        python_callable=run_forecast,
    )

    run_forecast_task
