import os
import sys

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

BRASILIO_PATH="/home/ivo/Documentos/brasilio/BrasilIO-1/"
if BRASILIO_PATH not in sys.path:
    sys.path.append(BRASILIO_PATH)

from jobs.download_raw import run as download_raw_run
from jobs.raw_para_bronze import run as raw_para_bronze_run
from jobs.bronze_para_silver import run as bronze_para_silver_run
from jobs.silver_para_gold_duck import run as silver_para_gold_duck_run
   
with DAG(
    dag_id="brasilio_etl",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",  # ou @monthly, etc.
    catchup=False,
    tags=["brasil-io", "engenharia-dados"],
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
