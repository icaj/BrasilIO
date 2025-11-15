import logging
from pathlib import Path
from typing import Any
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import numpy as np

from bronze import Bronze_Dataset as br

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Silver_Dataset:
    
    # Configurações de diretórios
    dir_silver = ""
    brz = Any
    
    def __init__(self, brz: br, dir="./dataset/silver"):
        self.dir_silver = Path(dir)
        self.brz = brz

    def limpar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Realiza limpeza e padronização completa de TODAS as colunas."""
        logging.info("Iniciando limpeza dos dados...")
        logging.info(f"Colunas encontradas: {list(df.columns)}")
        
        # 1. TRATAMENTO DE TODAS AS COLUNAS DE TEXTO/OBJECT
        colunas_texto = df.select_dtypes(include=["object"]).columns
        for col in colunas_texto:
            logging.info(f"Limpando coluna de texto: {col}")
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Remove valores 'nan', 'none', etc que viram string
            df[col] = df[col].replace(['nan', 'none', ''], pd.NA)
        
        # 2. TRATAMENTO DE TODAS AS COLUNAS NUMÉRICAS
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        for col in colunas_numericas:
            logging.info(f"Limpando coluna numérica: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Preenche nulos com 0 para colunas numéricas
            df[col] = df[col].fillna(0)
        
        # 3. TRATAMENTO DE TODAS AS COLUNAS DE DATA/DATETIME
        colunas_datetime = df.select_dtypes(include=['datetime64']).columns
        for col in colunas_datetime:
            logging.info(f"Convertendo data para texto: {col}")
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
                        logging.info(f"Coluna '{col}' detectada como data. Convertendo para texto...")
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
            logging.info(f"{linhas_removidas} linhas duplicadas removidas")
        
        logging.info("Limpeza concluída.")
        logging.info(f"Total de registros após limpeza: {len(df)}")
        return df


    def testes_qualidade(self, df: pd.DataFrame) -> None:
        """Executa testes de qualidade em TODAS as colunas."""
        logging.info("Executando testes de qualidade...")
        
        # Testa todas as colunas
        logging.info("Relatório de Valores Nulos por Coluna:")
        logging.info("-" * 60)
        for col in df.columns:
            nulos = df[col].isna().sum()
            percentual = (nulos / len(df)) * 100
            if nulos > 0:
                logging.info(f"  {col}: {nulos} nulos ({percentual:.2f}%)")
            else:
                logging.info(f" {col}: sem valores nulos")
        
        # Colunas críticas (se existirem)
        colunas_criticas = ["ano", "mes", "valor", "data_pagamento"]
        logging.info("Análise de Colunas Críticas:")
        logging.info("-" * 60)
        for col in colunas_criticas:
            if col in df.columns:
                nulos = df[col].isna().sum()
                if nulos > 0:
                    logging.warning(f" CRÍTICO: '{col}' possui {nulos} valores nulos")
                else:
                    logging.info(f" '{col}' OK")
            else:
                logging.warning(f"  Coluna crítica não encontrada: '{col}'")
        
        # Estatísticas gerais
        logging.info(f"Total de registros: {len(df)}")
        logging.info(f"Total de colunas: {len(df.columns)}")
        logging.info("Testes de qualidade concluídos.")

    def analise_exploratoria(self, df: pd.DataFrame) -> None:
        """Mostra estatísticas básicas e possíveis insights de negócio."""
        logging.info("="*60)
        logging.info("[ANÁLISE EXPLORATÓRIA]")
        logging.info("="*60)
        
        logging.info("Resumo estatístico de colunas numéricas:")
        logging.info(df.describe(include="number"))
        
        if "valor" in df.columns:
            total = df["valor"].sum()
            media = df["valor"].mean()
            mediana = df["valor"].median()
            logging.info(f"Análise de Valores:")
            logging.info(f"   • Total gasto: R$ {total:,.2f}")
            logging.info(f"   • Média por registro: R$ {media:,.2f}")
            logging.info(f"   • Mediana: R$ {mediana:,.2f}")
        
        if "ano" in df.columns and "valor" in df.columns:
            logging.info("Gastos por ano:")
            gastos_ano = df.groupby("ano")["valor"].agg(['sum', 'count', 'mean'])
            gastos_ano.columns = ['Total', 'Qtd Registros', 'Média']
            logging.info(gastos_ano)
        
        if "mes" in df.columns and "valor" in df.columns:
            logging.info("Gastos por mês:")
            gastos_mes = df.groupby("mes")["valor"].agg(['sum', 'count'])
            gastos_mes.columns = ['Total', 'Qtd Registros']
            logging.info(gastos_mes)


    def processar_bronze_para_silver(self, dataset_name: str = "gastos-diretos") -> None:
        """Pipeline completo: lê parquet da Bronze, limpa, valida e salva em Silver."""
        bronze_path = self.brz.path() / dataset_name
        silver_path = self.dir_silver / dataset_name
        
        logging.info("="*60)
        logging.info("INICIANDO PIPELINE BRONZE → SILVER")
        logging.info("="*60)
        
        if not bronze_path.exists():
            logging.info(f"[ERRO] Pasta bronze não encontrada: {bronze_path}")
            return
        
        logging.info(f"Lendo dados da camada Bronze...")
        logging.info(f"   Origem: {bronze_path}")
        
        try:
            dataset = ds.dataset(bronze_path, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            logging.info(f"{len(df)} registros carregados da Bronze")
        except Exception as e:
            logging.info(f"[ERRO] Falha ao ler dados: {str(e)}")
            return
        
        # Pipeline de transformação
        df = self.limpar_dados(df)
        self.testes_qualidade(df)
        self.analise_exploratoria(df)
        
        # Salvar na Silver
        silver_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Gravando dados limpos na camada Silver...")
        logging.info(f"   Destino: {silver_path}")
        
        try:
            # Verifica se existem colunas de particionamento
            particoes = []
            if "ano" in df.columns:
                particoes.append("ano")
            if "mes" in df.columns:
                particoes.append("mes")
            
            if particoes:
                logging.info(f"   Particionando por: {', '.join(particoes)}")
                ds.write_dataset(
                    data=pa.Table.from_pandas(df),
                    base_dir=str(silver_path),
                    format="parquet",
                    partitioning=particoes,
                    existing_data_behavior="overwrite_or_ignore"
                )
            else:
                logging.info("   Salvando sem particionamento")
                ds.write_dataset(
                    data=pa.Table.from_pandas(df),
                    base_dir=str(silver_path),
                    format="parquet",
                    existing_data_behavior="overwrite_or_ignore"
                )
            
            logging.warning("Dados salvos na camada Silver com sucesso!")
            logging.info("="*60)
        except Exception as e:
            logging.error(f"Falha ao salvar dados: {str(e)}")

