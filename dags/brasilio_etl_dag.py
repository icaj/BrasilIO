from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from jobs.collect_raw import run as collect_raw_run
from jobs.raw_to_bronze import run as raw_to_bronze_run
from jobs.bronze_to_silver import run as bronze_to_silver_run
from jobs.silver_to_gold_duck import run as silver_to_gold_duck_run

with DAG(
    dag_id="brasilio_etl",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",  # ou @monthly, etc.
    catchup=False,
    tags=["brasil-io", "engenharia-dados"],
) as dag:

    t_raw = PythonOperator(
        task_id="collect_raw",
        python_callable=collect_raw_run,
    )

    t_bronze = PythonOperator(
        task_id="raw_to_bronze",
        python_callable=raw_to_bronze_run,
    )

    t_silver = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=bronze_to_silver_run,
    )

    t_gold_duck = PythonOperator(
        task_id="silver_to_gold_duck",
        python_callable=silver_to_gold_duck_run,
    )

    # Orquestração
    t_raw >> t_bronze >> t_silver >> t_gold_duck
