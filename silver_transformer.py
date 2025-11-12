# silver_transformer.py
import os
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa

# Configurações de diretórios
BASE_DIR = Path("./dataset")
DIR_BRONZE = BASE_DIR / "bronze"
DIR_SILVER = BASE_DIR / "silver"

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza limpeza e padronização básica dos dados."""
    print("[INFO] Iniciando limpeza dos dados...")

    # Remover espaços extras e padronizar strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Conversões de tipos
    if "data_pagamento" in df.columns:
        df["data_pagamento"] = pd.to_datetime(df["data_pagamento"], errors="coerce")

    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)

    if "ano" in df.columns:
        df["ano"] = df["ano"].fillna(df["data_pagamento"].dt.year)
        df["ano"] = df["ano"].astype("Int64")

    if "mes" in df.columns:
        df["mes"] = df["mes"].fillna(df["data_pagamento"].dt.month)
        df["mes"] = df["mes"].astype("Int64")

    # Remover duplicatas
    df = df.drop_duplicates()

    print("[INFO] Limpeza concluída.")
    return df


def testes_qualidade(df: pd.DataFrame) -> None:
    """Executa testes simples de qualidade nos dados."""
    print("[INFO] Executando testes de qualidade...")
    colunas_criticas = ["ano", "mes", "valor", "data_pagamento"]

    for col in colunas_criticas:
        if col in df.columns:
            nulos = df[col].isna().sum()
            if nulos > 0:
                print(f"[ALERTA] Coluna '{col}' possui {nulos} valores nulos.")
        else:
            print(f"[ERRO] Coluna crítica ausente: {col}")

    print("[INFO] Testes de qualidade concluídos.")


def analise_exploratoria(df: pd.DataFrame) -> None:
    """Mostra estatísticas básicas e possíveis insights de negócio."""
    print("\n[ANÁLISE EXPLORATÓRIA]")
    print("Resumo estatístico de colunas numéricas:")
    print(df.describe(include="number"))

    if "valor" in df.columns:
        total = df["valor"].sum()
        media = df["valor"].mean()
        print(f"\nValor total gasto: R$ {total:,.2f}")
        print(f"Valor médio por registro: R$ {media:,.2f}")

    if "ano" in df.columns:
        print("\nGastos por ano:")
        print(df.groupby("ano")["valor"].sum())


def processar_bronze_para_silver(dataset_name: str = "gastos-diretos") -> None:
    """Pipeline completo: lê parquet da Bronze, limpa, valida e salva em Silver."""
    bronze_path = DIR_BRONZE / dataset_name
    silver_path = DIR_SILVER / dataset_name

    if not bronze_path.exists():
        print(f"[ERRO] Pasta bronze não encontrada: {bronze_path}")
        return

    print(f"[INFO] Lendo dados da camada Bronze ({bronze_path})...")
    dataset = ds.dataset(bronze_path, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()

    print(f"[INFO] {len(df)} registros carregados da Bronze")

    df = limpar_dados(df)
    testes_qualidade(df)
    analise_exploratoria(df)

    silver_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Gravando dados limpos na camada Silver ({silver_path})...")

    # Grava os arquivos particionados
    ds.write_dataset(
        data=pa.Table.from_pandas(df),
        base_dir=str(silver_path),
        format="parquet",
        partitioning=["ano", "mes"],
        existing_data_behavior="overwrite_or_ignore"
    )

    print("[SUCESSO] Dados salvos na camada Silver com sucesso.")


if __name__ == "__main__":
    processar_bronze_para_silver()
