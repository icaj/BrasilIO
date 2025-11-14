import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import pandas as pd

from raw import Raw_Dataset as rw

class Bronze_Dataset:

    DIR_BRONZE     = Path("./dataset/bronze")
    DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw            = Any

    def __init__(Self, raw: rw, dir="./dataset/bronze"):
        Self.DIR_BRONZE = Path(dir)
        Self.raw = raw
        
    def path(Self) -> Path:
        return Self.DIR_BRONZE
    
    def grava_parquet_particionado(Self, df: pd.DataFrame, nome_dataset: str = "gastos-diretos") -> List[Path]:
        """
        Grava DataFrame em arquivos Parquet particionados por ano e mês
        
        Args:
            df: DataFrame com os dados
            nome_dataset: Nome base do dataset
        
        Returns:
            Lista com os caminhos dos arquivos gerados
        """
        if df.empty:
            print("[AVISO] DataFrame vazio, nada a gravar")
            return []
        
        # Verifica se as colunas de partição existem
        if 'ano' not in df.columns or 'mes' not in df.columns:
            print("[ERRO] DataFrame não possui colunas 'ano' e 'mes' para particionamento")
            return []
        
        paths: List[Path] = []
        
        # Remove registros com ano ou mês nulos
        df_valido = df.dropna(subset=['ano', 'mes'])
        
        if len(df_valido) < len(df):
            print(f"[AVISO] {len(df) - len(df_valido)} registros removidos por falta de ano/mês")
        
        # Agrupa por ano e mês
        for (ano, mes), grupo_df in df_valido.groupby(['ano', 'mes']):
            # Formata mes com dois dígitos
            mes_fmt = f"{int(mes):02d}"
            
            # Cria o diretório da partição
            partition_dir = Self.DIR_BRONZE / nome_dataset / f"ano={int(ano)}" / f"mes={mes_fmt}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo
            arquivo = partition_dir / f"{nome_dataset}_ano{int(ano)}_mes{mes_fmt}.parquet"
            
            # Remove as colunas de partição do DataFrame (opcional)
            # grupo_df = grupo_df.drop(columns=['ano', 'mes'], errors='ignore')
            
            # Grava o parquet
            grupo_df.to_parquet(arquivo, index=False, engine='pyarrow')
            
            print(f"[INFO] Gravado: {arquivo} ({len(grupo_df)} registros)")
            paths.append(arquivo)
        
        return paths

    def transforma_json_para_df(Self, dados: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforma dados JSON em DataFrame
        
        Args:
            dados: Dicionário com os dados JSON
        
        Returns:
            DataFrame com os dados normalizados
        """
        # Extrai os itens (results ou data)
        itens = dados.get("results")
        if itens is None:
            itens = dados.get("data", dados if isinstance(dados, list) else [])
        
        if not itens:
            print("[AVISO] Não há dados para processar")
            return pd.DataFrame()
        
        # Normaliza o JSON em DataFrame
        df = pd.json_normalize(itens, max_level=1)
        
        # Converte a coluna data_pagamento para datetime
        if 'data_pagamento' in df.columns:
            df['data_pagamento'] = pd.to_datetime(df['data_pagamento'], errors='coerce')
        
        # Converte valor para float
        if 'valor' in df.columns:
            df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        
        # Garante que ano e mes existem e são inteiros
        if 'ano' in df.columns:
            df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
        
        if 'mes' in df.columns:
            df['mes'] = pd.to_numeric(df['mes'], errors='coerce').astype('Int64')
        
        print(f"[INFO] DataFrame criado com {len(df)} registros")
        return df

    def processar_json_para_parquet(Self, nome_arquivo: str, nome_dataset: str = "gastos-diretos") -> List[Path]:
        """
        Função principal que processa um arquivo JSON e gera os Parquets particionados
        
        Args:
            nome_arquivo: Nome do arquivo JSON na pasta raw
            nome_dataset: Nome do dataset
        
        Returns:
            Lista com os caminhos dos arquivos gerados
        """
        print(f"[INFO] Processando arquivo: {nome_arquivo}")
        
        # 1. Lê o JSON da pasta raw
        dados = Self.raw.le_json_raw(nome_arquivo)
        print(f"[INFO] Total de registros no JSON: {dados.get('count', 'N/A')}")
        
        # 2. Transforma em DataFrame
        df = Self.transforma_json_para_df(dados)
        
        if df.empty:
            print("[INFO] Nenhum dado para processar")
            return []
        
        # 3. Grava os Parquets particionados
        arquivos = Self.grava_parquet_particionado(df, nome_dataset)
        
        print(f"\n[SUCESSO] {len(arquivos)} arquivo(s) Parquet gerado(s)")
        return arquivos

    def transformar_raw_para_bronze(Self) -> int:
        try:
            arquivos = Self.raw.listdir()
            for arquivo in arquivos:
                arqs = Self.processar_json_para_parquet(arquivo)
            return len(arqs)
        except Exception:
            return 0
        
