import sys
from pathlib import Path

from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
from jobs.airflow_download_raw import run as download_raw_run
from jobs.airflow_raw_para_bronze import run as raw_para_bronze_run
from jobs.airflow_bronze_para_silver import run as bronze_para_silver_run
from jobs.airflow_silver_para_gold_duck import run as silver_para_gold_duck_run
   
with DAG(
    dag_id="0_brasilio_etl",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",  # ou @monthly, etc.
    catchup=False,
    tags=["cesar.school", "brasil-io", "engenharia-dados"],
) as dag:

    t_raw = PythonOperator(
        task_id="download_raw",
        python_callable=download_raw_run,
    )

    t_bronze = PythonOperator(
        task_id="raw_para_bronze",
        python_callable=raw_para_bronze_run,
    )

    t_silver = PythonOperator(
        task_id="bronze_para_silver",
        python_callable=bronze_para_silver_run,
    )

    t_gold_duck = PythonOperator(
        task_id="silver_para_gold_duck",
        python_callable=silver_para_gold_duck_run,
    )

    # Orquestração
    t_raw >> t_bronze >> t_silver >> t_gold_duck
