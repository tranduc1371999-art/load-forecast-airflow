FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY jobs ./jobs

CMD ["python", "jobs/run_forecast.py"]