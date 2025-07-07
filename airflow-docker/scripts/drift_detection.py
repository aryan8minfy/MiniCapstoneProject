from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

def run_drift_reports():
    df = pd.read_csv("/opt/airflow/data/processed/cleaned.csv")
    TARGET = 'adjusted_total_usd'
    X = df.drop(columns=[TARGET])

    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X_train, current_data=X_test)
    report.save_html("/opt/airflow/reports/data_drift.html")
