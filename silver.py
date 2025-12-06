import logging
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from bronze import Bronze_Dataset as br

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Schema da camada Silver (fixo e consistente)
SILVER_SCHEMA = pa.schema([
    pa.field("ano", pa.int32()),
    pa.field("codigo_acao", pa.int32()),
    pa.field("codigo_elemento_despesa", pa.int32()),
    pa.field("codigo_favorecido", pa.string()),
    pa.field("codigo_funcao", pa.int32()),
    pa.field("codigo_grupo_despesa", pa.int32()),
    pa.field("codigo_orgao", pa.int32()),
    pa.field("codigo_orgao_superior", pa.int32()),
    pa.field("codigo_programa", pa.int32()),
    pa.field("codigo_subfuncao", pa.int32()),
    pa.field("codigo_unidade_gestora", pa.int32()),
    pa.field("data_pagamento", pa.timestamp("ns")),
    pa.field("data_pagamento_original", pa.string()),
    pa.field("gestao_pagamento", pa.string()),
    pa.field("linguagem_cidada", pa.string()),
    pa.field("mes", pa.int32()),
    pa.field("nome_acao", pa.string()),
    pa.field("nome_elemento_despesa", pa.string()),
    pa.field("nome_favorecido", pa.string()),
    pa.field("nome_funcao", pa.string()),
    pa.field("nome_grupo_despesa", pa.string()),
    pa.field("nome_orgao", pa.string()),
    pa.field("nome_orgao_superior", pa.string()),
    pa.field("nome_programa", pa.string()),
    pa.field("nome_subfuncao", pa.string()),
    pa.field("nome_unidade_gestora", pa.string()),
    pa.field("numero_documento", pa.string()),
    pa.field("valor", pa.float64()),
])


class Silver_Dataset:
    """
    Camada Silver:
    - Lê Parquets da Bronze (particionados por ano/mes).
    - Faz limpeza e padronização.
    - Garante schema consistente (SILVER_SCHEMA).
    - Escreve Parquets na pasta dataset/silver/<dataset_name>, particionados por ano/mes.
    """

    dir_silver: Path
    brz: Any

    def __init__(self, brz: br, dir: str = "./dataset/silver") -> None:
        self.dir_silver = Path(dir)
        self.brz = brz

    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------
    def path(self) -> Path:
        return self.dir_silver

    def listar_periodos_disponiveis(self, bronze_path: Path) -> list[tuple[str, str]]:
        """
        Lista todos os anos e meses disponíveis na camada Bronze.
        Estrutura esperada: bronze/<dataset>/ano=XXXX/mes=XX/*.parquet
        """
        periodos: List[tuple[str, str]] = []

        if not bronze_path.exists():
            return periodos

        # Estrutura particionada ano=XXXX/mes=XX
        for ano_dir in bronze_path.glob("ano=*"):
            ano = ano_dir.name.split("=")[1]
            for mes_dir in ano_dir.glob("mes=*"):
                mes = mes_dir.name.split("=")[1]
                if list(mes_dir.glob("*.parquet")):
                    periodos.append((ano, mes))

        # Caso não tenha partição, usa arquivos diretos
        if not periodos and list(bronze_path.glob("*.parquet")):
            periodos.append(("sem_particao", "sem_particao"))

        return sorted(periodos)

    # -------------------------------------------------------------------------
    # Limpeza e qualidade
    # -------------------------------------------------------------------------
    def limpar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza limpeza e padronização de TODAS as colunas.
        Deixa a base pronta para aplicar o SILVER_SCHEMA.
        """
        logging.info("Iniciando limpeza dos dados...")
        logging.info(f"Colunas encontradas: {list(df.columns)}")

        df = df.copy()

        # 1. Colunas de texto: strip + lower + tratamento de 'nan', 'none'
        colunas_texto = df.select_dtypes(include=["object"]).columns
        for col in colunas_texto:
            logging.info(f"Limpando coluna de texto: {col}")
            df[col] = df[col].astype(str).str.strip()
            # lower só nas colunas que não são "data_pagamento_original"/"numero_documento"
            if col not in ["data_pagamento_original", "numero_documento"]:
                df[col] = df[col].str.lower()

            df[col] = df[col].replace(["nan", "none", ""], pd.NA)

        # 2. Colunas numéricas existentes: garantir numeric
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        for col in colunas_numericas:
            logging.info(f"Limpando coluna numérica: {col}")
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Data de pagamento (se existir)
        #    - preserva data_pagamento_original como string
        #    - data_pagamento como datetime64[ns]
        if "data_pagamento" in df.columns:
            logging.info("Convertendo 'data_pagamento' para datetime...")
            df["data_pagamento_original"] = df["data_pagamento"].astype(str)
            df["data_pagamento"] = pd.to_datetime(
                df["data_pagamento"],
                errors="coerce",
                utc=False,
            )

        # 4. Campos ano/mes: se não existirem ou estiverem nulos, derivar da data_pagamento
        if "ano" not in df.columns and "data_pagamento" in df.columns:
            df["ano"] = df["data_pagamento"].dt.year

        if "mes" not in df.columns and "data_pagamento" in df.columns:
            df["mes"] = df["data_pagamento"].dt.month

        # 5. Tratamento de tipos finais (inteiros/float)
        int_cols = [
            "ano",
            "codigo_acao",
            "codigo_elemento_despesa",
            "codigo_funcao",
            "codigo_grupo_despesa",
            "codigo_orgao",
            "codigo_orgao_superior",
            "codigo_programa",
            "codigo_subfuncao",
            "codigo_unidade_gestora",
            "mes",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

        if "valor" in df.columns:
            df["valor"] = pd.to_numeric(df["valor"], errors="coerce").astype("float64")
        else:
            df["valor"] = np.nan

        # 6. Remover linhas totalmente vazias
        df = df.dropna(how="all")

        # 7. Remover duplicadas
        linhas_antes = len(df)
        df = df.drop_duplicates()
        linhas_removidas = linhas_antes - len(df)
        if linhas_removidas > 0:
            logging.info(f"{linhas_removidas} linhas duplicadas removidas")

        logging.info("Limpeza concluída.")
        logging.info(f"Total de registros após limpeza: {len(df)}")

        return df

    def testes_qualidade(self, df: pd.DataFrame) -> None:
        """Executa testes de qualidade em colunas relevantes."""
        logging.info("Executando testes de qualidade...")

        logging.info("Relatório de Valores Nulos por Coluna:")
        logging.info("-" * 60)
        for col in df.columns:
            nulos = df[col].isna().sum()
            percentual = (nulos / len(df)) * 100 if len(df) > 0 else 0
            if nulos > 0:
                logging.info(f" {col}: {nulos} nulos ({percentual:.2f}%)")
            else:
                logging.info(f" {col}: sem valores nulos")

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
                logging.warning(f" Coluna crítica não encontrada: '{col}'")

        logging.info(f"Total de registros: {len(df)}")
        logging.info(f"Total de colunas: {len(df.columns)}")
        logging.info("Testes de qualidade concluídos.")

    def analise_exploratoria(self, df: pd.DataFrame) -> None:
        """Estatísticas básicas e alguns insights rápidos."""
        logging.info("=" * 60)
        logging.info("[ANÁLISE EXPLORATÓRIA]")
        logging.info("=" * 60)

        logging.info("Resumo estatístico de colunas numéricas:")
        logging.info(df.describe(include="number"))

        if "valor" in df.columns:
            total = df["valor"].sum()
            media = df["valor"].mean()
            mediana = df["valor"].median()
            logging.info("Análise de Valores:")
            logging.info(f" • Total gasto: R$ {total:,.2f}")
            logging.info(f" • Média por registro: R$ {media:,.2f}")
            logging.info(f" • Mediana: R$ {mediana:,.2f}")

        if "ano" in df.columns and "valor" in df.columns:
            logging.info("Gastos por ano:")
            gastos_ano = df.groupby("ano")["valor"].agg(["sum", "count", "mean"])
            gastos_ano.columns = ["Total", "Qtd Registros", "Média"]
            logging.info("\n%s", gastos_ano)

        if "mes" in df.columns and "valor" in df.columns:
            logging.info("Gastos por mês:")
            gastos_mes = df.groupby("mes")["valor"].agg(["sum", "count"])
            gastos_mes.columns = ["Total", "Qtd Registros"]
            logging.info("\n%s", gastos_mes)

    # -------------------------------------------------------------------------
    # Pipeline Bronze -> Silver
    # -------------------------------------------------------------------------
    def processar_bronze_para_silver(self, dataset_name: str = "gastos-diretos") -> None:
        """
        Pipeline completo:
        - Lê todos os Parquets da Bronze
        - Limpa / padroniza
        - Garante schema SILVER_SCHEMA
        - Grava Parquets particionados por ano/mes na Silver
        """
        bronze_path = self.brz.path() / dataset_name
        silver_path = self.dir_silver / dataset_name

        logging.info("=" * 80)
        logging.info(
            "INICIANDO PIPELINE BRONZE %s → SILVER %s",
            self.brz.dir_bronze,
            self.dir_silver,
        )
        logging.info("=" * 80)

        if not bronze_path.exists():
            logging.error("[ERRO] Pasta bronze não encontrada: %s", bronze_path)
            return

        periodos = self.listar_periodos_disponiveis(bronze_path)
        if not periodos:
            logging.error(
                "[ERRO] Nenhum arquivo parquet encontrado em: %s",
                bronze_path,
            )
            return

        logging.info(" Períodos encontrados na Bronze: %d", len(periodos))
        for ano, mes in periodos:
            if ano != "sem_particao":
                logging.info(" • Ano %s / Mês %s", ano, mes)
            else:
                logging.info(" • Dados sem particionamento")

        logging.info("-" * 60)
        logging.info("Lendo TODOS os dados da camada Bronze...")
        logging.info(" Origem: %s", bronze_path)

        try:
            dataset = ds.dataset(bronze_path, format="parquet", partitioning="hive", schema=SILVER_SCHEMA)
            table = dataset.to_table()
            df = table.to_pandas()
            logging.info(" %d registros carregados da Bronze", len(df))

            if "ano" in df.columns and "mes" in df.columns:
                logging.info(" Distribuição de registros por período:")
                distribuicao = (
                    df.groupby(["ano", "mes"])
                    .size()
                    .reset_index(name="registros")
                )
                for _, row in distribuicao.iterrows():
                    logging.info(
                        " • %s/%02d: %d registros",
                        str(row["ano"]),
                        int(row["mes"]),
                        int(row["registros"]),
                    )
        except Exception as e:
            logging.error("[ERRO] Falha ao ler dados: %s", str(e))
            return

        # Limpeza + QA + análise rápida
        df = self.limpar_dados(df)
        self.testes_qualidade(df)
        self.analise_exploratoria(df)

        # Garante que TODAS as colunas do schema existam
        for field in SILVER_SCHEMA:
            if field.name not in df.columns:
                if pa.types.is_integer(field.type):
                    df[field.name] = pd.Series(dtype="Int32")
                elif pa.types.is_floating(field.type):
                    df[field.name] = pd.Series(dtype="float64")
                elif pa.types.is_timestamp(field.type):
                    df[field.name] = pd.NaT
                else:
                    df[field.name] = pd.Series(dtype="object")

        # Reordena colunas na ordem do schema
        df = df[[f.name for f in SILVER_SCHEMA]]

        # Ajuste final de tipos para bater com o schema
        int_cols = [
            "ano",
            "codigo_acao",
            "codigo_elemento_despesa",
            "codigo_funcao",
            "codigo_grupo_despesa",
            "codigo_orgao",
            "codigo_orgao_superior",
            "codigo_programa",
            "codigo_subfuncao",
            "codigo_unidade_gestora",
            "mes",
        ]
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

        df["valor"] = pd.to_numeric(df["valor"], errors="coerce").astype("float64")
        df["data_pagamento"] = pd.to_datetime(
            df["data_pagamento"],
            errors="coerce",
            utc=False,
        )

        # Escrever na Silver
        silver_path.mkdir(parents=True, exist_ok=True)
        logging.info("Gravando dados limpos na camada Silver...")
        logging.info(" Destino: %s", silver_path)

        try:
            table = pa.Table.from_pandas(
                df,
                schema=SILVER_SCHEMA,
                preserve_index=False,
                safe=False,  # permite cast automático para o schema
            )

            partition_cols = []
            if "ano" in df.columns:
                partition_cols.append("ano")
            if "mes" in df.columns:
                partition_cols.append("mes")

            if partition_cols:
                logging.info(" Particionando por: %s", ", ".join(partition_cols))
                ds.write_dataset(
                    data=table,
                    base_dir=str(silver_path),
                    format="parquet",
                    partitioning=partition_cols,
                    existing_data_behavior="overwrite_or_ignore",
                )
            else:
                logging.info(" Salvando sem particionamento")
                ds.write_dataset(
                    data=table,
                    base_dir=str(silver_path),
                    format="parquet",
                    existing_data_behavior="overwrite_or_ignore",
                )

            logging.info(" Dados salvos na camada Silver com sucesso!")
            logging.info(" Total de registros processados: %d", len(df))
            logging.info(" Total de períodos processados: %d", len(periodos))
            logging.info("=" * 60)

        except Exception as e:
            logging.error(" Falha ao salvar dados: %s", str(e))
