# gold_transformer.py
import os
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from streamlit import streamlit

# Configurações
BASE_DIR = Path("./dataset")
DIR_SILVER = BASE_DIR / "silver"
DIR_GOLD = BASE_DIR / "gold"

def criar_agregacoes(df: pd.DataFrame) -> dict:
    """
    Cria tabelas agregadas a partir dos dados da camada Silver.
    Retorna um dicionário de DataFrames nomeados.
    """
    print("[INFO] Criando agregações para camada Gold...")

    agregacoes = {}

    # 1. Total de gastos por órgão e mês
    if {"orgao", "ano", "mes", "valor"}.issubset(df.columns):
        gastos_orgao = (
            df.groupby(["orgao", "ano", "mes"], dropna=False)["valor"]
            .sum()
            .reset_index()
            .sort_values(["ano", "mes", "valor"], ascending=[True, True, False])
        )
        agregacoes["gastos_por_orgao_mes"] = gastos_orgao

    # 2. Total de gastos por categoria de despesa (se existir)
    for col_cat in ["favorecido", "subfuncao", "elemento_despesa"]:
        if col_cat in df.columns:
            agg = (
                df.groupby([col_cat, "ano", "mes"], dropna=False)["valor"]
                .sum()
                .reset_index()
            )
            agregacoes[f"gastos_por_{col_cat}_mes"] = agg

    # 3. Gastos anuais totais
    if {"ano", "valor"}.issubset(df.columns):
        gastos_anuais = df.groupby("ano", dropna=False)["valor"].sum().reset_index()
        agregacoes["gastos_totais_ano"] = gastos_anuais

    print(f"[INFO] {len(agregacoes)} agregações criadas.")
    return agregacoes


def salvar_agregacoes(agregacoes: dict, dataset_name: str = "gastos-diretos"):
    """Salva as tabelas agregadas no diretório Gold."""
    gold_path = DIR_GOLD / dataset_name
    gold_path.mkdir(parents=True, exist_ok=True)

    for nome, df in agregacoes.items():
        caminho = gold_path / f"{nome}.parquet"
        print(f"[INFO] Salvando {nome} → {caminho}")
        table = pa.Table.from_pandas(df)
        pa.parquet.write_table(table, caminho)

    print(f"[SUCESSO] Dados salvos na camada Gold em {gold_path}")


def analise_gold(df: pd.DataFrame):
    """Exibe alguns insights da camada Gold."""
    print("\n[ANÁLISE GOLD]")
    if "valor" in df.columns:
        print(f"Valor total gasto: R$ {df['valor'].sum():,.2f}")
        print(f"Gasto médio: R$ {df['valor'].mean():,.2f}")

    if {"ano", "valor"}.issubset(df.columns):
        print("\nGasto anual (Top 5 anos):")
        print(df.groupby("ano")["valor"].sum().nlargest(5))


def processar_silver_para_gold(dataset_name: str = "gastos-diretos"):
    """Pipeline da camada Silver → Gold."""
    silver_path = DIR_SILVER / dataset_name
    if not silver_path.exists():
        print(f"[ERRO] Pasta Silver não encontrada: {silver_path}")
        return

    print(f"[INFO] Lendo dados da camada Silver ({silver_path})...")
    dataset = ds.dataset(silver_path, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()

    print(f"[INFO] {len(df)} registros carregados da Silver.")

    # Cria agregações
    agregacoes = criar_agregacoes(df)

    # Salva resultados
    salvar_agregacoes(agregacoes, dataset_name=dataset_name)

    # Mostra análise exploratória Gold
    analise_gold(df)


#if __name__ == "__main__":
#    processar_silver_para_gold()
