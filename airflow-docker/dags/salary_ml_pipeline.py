import sys
sys.path.insert(0, '/opt/airflow') 

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from scripts.data_cleaning import run_cleaning
from scripts.model_training import run_model_training
from scripts.drift_detection import run_drift_reports
from scripts.shap_explainer import run_shap_explainer

with DAG(
    dag_id="salary_prediction_pipeline",
    description="Train model, detect drift, and explain predictions weekly",
    start_date=datetime(2025, 7, 6),
    schedule_interval="@weekly",
    catchup=False
) as dag:

    clean_data = PythonOperator(task_id="clean_data", python_callable=run_cleaning)
    train_model = PythonOperator(task_id="train_model", python_callable=run_model_training)
    detect_drift = PythonOperator(task_id="detect_drift", python_callable=run_drift_reports)
    explain_model = PythonOperator(task_id="explain_model", python_callable=run_shap_explainer)

    clean_data >> train_model >> detect_drift >> explain_model
