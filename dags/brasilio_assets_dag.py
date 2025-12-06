# dags/brasilio_etl_assets_dag.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.datasets import Dataset

# 🔹 Ajuste este caminho para o path do projeto BrasilIO *dentro do worker*
BRASILIO_PATH = "/home/ivo/Documentos/brasilio/BrasilIO-1/"  # EXEMPLO: ajuste para o seu ambiente

if BRASILIO_PATH not in sys.path:
    sys.path.append(BRASILIO_PATH)

# Imports dos jobs já existentes no repo
from jobs.download_raw import run as download_raw_run
from jobs.raw_para_bronze import run as raw_para_bronze_run
from jobs.bronze_para_silver import run as bronze_para_silver_run
from jobs.silver_para_gold_duck import run as silver_para_gold_duck_run

# 🔹 Definição dos Datasets (Assets) – vão aparecer no menu "Assets"
DATASET_NAME = "gastos-diretos"

RAW_DATASET = Dataset(f"brasilio://raw/{DATASET_NAME}")
BRONZE_DATASET = Dataset(f"brasilio://bronze/{DATASET_NAME}")
SILVER_DATASET = Dataset(f"brasilio://silver/{DATASET_NAME}")
GOLD_DATASET = Dataset(f"brasilio://gold/{DATASET_NAME}")
DUCKDB_DATASET = Dataset(f"brasilio://duckdb/{DATASET_NAME}")

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="brasilio_etl_assets",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",          # roda de hora em hora
    catchup=False,               # não faz backfill automático
    default_args=default_args,
    tags=["brasilio", "gastos-diretos", "assets", "duckdb"],
) as dag:

    # 1) RAW – baixa páginas da API e grava em dataset/raw (JSON)
    t_raw = PythonOperator(
        task_id="download_raw",
        python_callable=download_raw_run,
        # 🔹 este task PRODUZ o asset RAW
        outlets=[RAW_DATASET],
    )

    # 2) BRONZE – transforma JSON → Parquet particionado ano/mês (dataset/bronze)
    t_bronze = PythonOperator(
        task_id="raw_para_bronze",
        python_callable=raw_para_bronze_run,
        # 🔹 consome RAW (conceitualmente) e produz BRONZE
        outlets=[BRONZE_DATASET],
    )

    # 3) SILVER – limpa e padroniza (schema fixo SILVER_SCHEMA em silver.py)
    t_silver = PythonOperator(
        task_id="bronze_para_silver",
        python_callable=bronze_para_silver_run,
        outlets=[SILVER_DATASET],
    )

    # 4) GOLD + DuckDB – gera agregações Gold + views/tabelas no DuckDB
    t_gold_duck = PythonOperator(
        task_id="silver_para_gold_duck",
        python_callable=silver_para_gold_duck_run,
        # 🔹 aqui você está:
        #   - escrevendo Parquets em dataset/gold
        #   - atualizando o banco DuckDB (views/tabelas temporais)
        outlets=[GOLD_DATASET, DUCKDB_DATASET],
    )

    # Orquestração em cadeia – cada task só roda se a anterior tiver SUCCESS
    t_raw >> t_bronze >> t_silver >> t_gold_duck
