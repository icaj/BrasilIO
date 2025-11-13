# silver_transformer.py
import os
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import numpy as np

# Configurações de diretórios
BASE_DIR = Path("./dataset")
DIR_BRONZE = BASE_DIR / "bronze"
DIR_SILVER = BASE_DIR / "silver"

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Realiza limpeza e padronização completa de TODAS as colunas."""
    print("[INFO] Iniciando limpeza dos dados...")
    print(f"[INFO] Colunas encontradas: {list(df.columns)}")
    
    # 1. TRATAMENTO DE TODAS AS COLUNAS DE TEXTO/OBJECT
    colunas_texto = df.select_dtypes(include=["object"]).columns
    for col in colunas_texto:
        print(f"[INFO] Limpando coluna de texto: {col}")
        df[col] = df[col].astype(str).str.strip().str.lower()
        # Remove valores 'nan', 'none', etc que viram string
        df[col] = df[col].replace(['nan', 'none', ''], pd.NA)
    
    # 2. TRATAMENTO DE TODAS AS COLUNAS NUMÉRICAS
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        print(f"[INFO] Limpando coluna numérica: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Preenche nulos com 0 para colunas numéricas
        df[col] = df[col].fillna(0)
    
    # 3. TRATAMENTO DE TODAS AS COLUNAS DE DATA/DATETIME
    colunas_datetime = df.select_dtypes(include=['datetime64']).columns
    for col in colunas_datetime:
        print(f"[INFO] Convertendo data para texto: {col}")
        # Converte datetime para string no formato ISO (YYYY-MM-DD)
        df[col] = df[col].dt.strftime('%Y-%m-%d')
        df[col] = df[col].replace('NaT', pd.NA)
    
    # 4. DETECÇÃO AUTOMÁTICA DE COLUNAS QUE PARECEM DATAS (mas estão como texto)
    for col in df.columns:
        if any(palavra in col.lower() for palavra in ['data', 'date', 'dt', 'dia']):
            try:
                # Tenta converter para datetime e depois para texto
                temp_date = pd.to_datetime(df[col], errors='coerce')
                if temp_date.notna().sum() > len(df) * 0.5:  # Se >50% são datas válidas
                    print(f"[INFO] Coluna '{col}' detectada como data. Convertendo para texto...")
                    df[col] = temp_date.dt.strftime('%Y-%m-%d')
                    df[col] = df[col].replace('NaT', pd.NA)
            except:
                pass
    
    # 5. TRATAMENTO ESPECIAL PARA COLUNAS CONHECIDAS
    # Ano e Mês
    if "ano" in df.columns:
        if "data_pagamento" in df.columns and df["data_pagamento"].notna().any():
            # Se data_pagamento existe, usa ela para preencher ano
            df["ano"] = df["ano"].fillna(pd.to_datetime(df["data_pagamento"]).dt.year)
        df["ano"] = df["ano"].astype("Int64")
    
    if "mes" in df.columns:
        if "data_pagamento" in df.columns and df["data_pagamento"].notna().any():
            df["mes"] = df["mes"].fillna(pd.to_datetime(df["data_pagamento"]).dt.month)
        df["mes"] = df["mes"].astype("Int64")
    
    # Valor
    if "valor" in df.columns:
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").fillna(0.0)
    
    # 6. REMOVER LINHAS COMPLETAMENTE VAZIAS
    df = df.dropna(how='all')
    
    # 7. REMOVER DUPLICATAS
    linhas_antes = len(df)
    df = df.drop_duplicates()
    linhas_removidas = linhas_antes - len(df)
    if linhas_removidas > 0:
        print(f"[INFO] {linhas_removidas} linhas duplicadas removidas")
    
    print("[INFO] Limpeza concluída.")
    print(f"[INFO] Total de registros após limpeza: {len(df)}")
    return df


def testes_qualidade(df: pd.DataFrame) -> None:
    """Executa testes de qualidade em TODAS as colunas."""
    print("\n[INFO] Executando testes de qualidade...")
    
    # Testa todas as colunas
    print("\n📊 Relatório de Valores Nulos por Coluna:")
    print("-" * 60)
    for col in df.columns:
        nulos = df[col].isna().sum()
        percentual = (nulos / len(df)) * 100
        if nulos > 0:
            print(f"⚠️  {col}: {nulos} nulos ({percentual:.2f}%)")
        else:
            print(f"✅ {col}: sem valores nulos")
    
    # Colunas críticas (se existirem)
    colunas_criticas = ["ano", "mes", "valor", "data_pagamento"]
    print("\n🔍 Análise de Colunas Críticas:")
    print("-" * 60)
    for col in colunas_criticas:
        if col in df.columns:
            nulos = df[col].isna().sum()
            if nulos > 0:
                print(f"❌ CRÍTICO: '{col}' possui {nulos} valores nulos")
            else:
                print(f"✅ '{col}' OK")
        else:
            print(f"⚠️  Coluna crítica não encontrada: '{col}'")
    
    # Estatísticas gerais
    print(f"\n📈 Total de registros: {len(df)}")
    print(f"📋 Total de colunas: {len(df.columns)}")
    print("[INFO] Testes de qualidade concluídos.")


def analise_exploratoria(df: pd.DataFrame) -> None:
    """Mostra estatísticas básicas e possíveis insights de negócio."""
    print("\n" + "="*60)
    print("[ANÁLISE EXPLORATÓRIA]")
    print("="*60)
    
    print("\n📊 Resumo estatístico de colunas numéricas:")
    print(df.describe(include="number"))
    
    if "valor" in df.columns:
        total = df["valor"].sum()
        media = df["valor"].mean()
        mediana = df["valor"].median()
        print(f"\n💰 Análise de Valores:")
        print(f"   • Total gasto: R$ {total:,.2f}")
        print(f"   • Média por registro: R$ {media:,.2f}")
        print(f"   • Mediana: R$ {mediana:,.2f}")
    
    if "ano" in df.columns and "valor" in df.columns:
        print("\n📅 Gastos por ano:")
        gastos_ano = df.groupby("ano")["valor"].agg(['sum', 'count', 'mean'])
        gastos_ano.columns = ['Total', 'Qtd Registros', 'Média']
        print(gastos_ano)
    
    if "mes" in df.columns and "valor" in df.columns:
        print("\n📆 Gastos por mês:")
        gastos_mes = df.groupby("mes")["valor"].agg(['sum', 'count'])
        gastos_mes.columns = ['Total', 'Qtd Registros']
        print(gastos_mes)


def processar_bronze_para_silver(dataset_name: str = "gastos-diretos") -> None:
    """Pipeline completo: lê parquet da Bronze, limpa, valida e salva em Silver."""
    bronze_path = DIR_BRONZE / dataset_name
    silver_path = DIR_SILVER / dataset_name
    
    print("="*60)
    print("🚀 INICIANDO PIPELINE BRONZE → SILVER")
    print("="*60)
    
    if not bronze_path.exists():
        print(f"❌ [ERRO] Pasta bronze não encontrada: {bronze_path}")
        return
    
    print(f"\n📂 Lendo dados da camada Bronze...")
    print(f"   Origem: {bronze_path}")
    
    try:
        dataset = ds.dataset(bronze_path, format="parquet")
        table = dataset.to_table()
        df = table.to_pandas()
        print(f"✅ {len(df)} registros carregados da Bronze")
    except Exception as e:
        print(f"❌ [ERRO] Falha ao ler dados: {str(e)}")
        return
    
    # Pipeline de transformação
    df = limpar_dados(df)
    testes_qualidade(df)
    analise_exploratoria(df)
    
    # Salvar na Silver
    silver_path.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Gravando dados limpos na camada Silver...")
    print(f"   Destino: {silver_path}")
    
    try:
        # Verifica se existem colunas de particionamento
        particoes = []
        if "ano" in df.columns:
            particoes.append("ano")
        if "mes" in df.columns:
            particoes.append("mes")
        
        if particoes:
            print(f"   Particionando por: {', '.join(particoes)}")
            ds.write_dataset(
                data=pa.Table.from_pandas(df),
                base_dir=str(silver_path),
                format="parquet",
                partitioning=particoes,
                existing_data_behavior="overwrite_or_ignore"
            )
        else:
            print("   Salvando sem particionamento")
            ds.write_dataset(
                data=pa.Table.from_pandas(df),
                base_dir=str(silver_path),
                format="parquet",
                existing_data_behavior="overwrite_or_ignore"
            )
        
        print("\n✅ [SUCESSO] Dados salvos na camada Silver com sucesso!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ [ERRO] Falha ao salvar dados: {str(e)}")


#if __name__ == "__main__":
#    processar_bronze_para_silver()