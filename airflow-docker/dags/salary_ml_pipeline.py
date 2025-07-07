from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta

# === Default Arguments ===
default_args = {
    'owner': 'aryan',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}

# === Function to Check Drift Flag ===
def check_drift_flag():
    flag_path = "/opt/airflow/flags/drift_detected.txt"
    try:
        with open(flag_path, "r") as f:
            drift_detected = f.read().strip().lower() == "yes"
            print(f"[INFO] Drift flag value: {drift_detected}")
            return drift_detected
    except FileNotFoundError:
        print("[WARNING] Drift flag not found.")
        return False

# === DAG Definition ===
with DAG(
    dag_id='salary_retrain_mlflow_pipeline',
    default_args=default_args,
    description='Minimal pipeline: detect drift and retrain salary model if needed',
    schedule_interval='@hourly',
    start_date=datetime(2025, 7, 1),
    catchup=False,
    tags=['mlops', 'salary', 'minimal'],
) as dag:

    # Task 1: Wait for new salary data
    sense_salary_data = FileSensor(
        task_id='sense_salary_data_file',
        filepath='/opt/airflow/data/new_salary_data.csv',
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    # Task 2: Run drift detection script
    task_run_drift_check = BashOperator(
        task_id='run_drift_check_script',
        bash_command='echo "[INFO] Running drift detection..." && python /opt/airflow/dags/detect_drift.py'
    )

    # Task 3: Evaluate drift flag
    task_check_drift_status = ShortCircuitOperator(
        task_id='evaluate_drift_status',
        python_callable=check_drift_flag
    )

    # Task 4: Retrain model if drift is found
    task_model_retraining = BashOperator(
        task_id='trigger_model_retraining',
        bash_command='echo "[INFO] Retraining salary model..." && python /opt/airflow/dags/train_model.py'
    )

    # Define Flow
    sense_salary_data >> task_run_drift_check >> task_check_drift_status >> task_model_retraining

