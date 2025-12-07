from __future__ import annotations
import logging
from pathlib import Path
from typing import List
import duckdb

# Configurações básicas
BASE_DIR = Path("./dataset")
BRONZE_DIR = BASE_DIR / "bronze"
SILVER_DIR = BASE_DIR / "silver"
GOLD_DIR = BASE_DIR / "gold"
DUCKDB_PATH = BASE_DIR / "brasilio.duckdb"

logger = logging.getLogger(__name__)

# Converte nomes em identificadores válidos para DuckDB:
#  - minúsculo
#  - troca '-', espaços, '/' e '\' por '_'
def sanear_nome(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

# Gerencia a integração com DuckDB:
# - Cria/abre o arquivo dataset/brasilio.duckdb
# - Cria views apontando para os Parquets da Bronze e Silver
# - Carrega Parquets da Gold como tabelas
# - Cria tabelas de séries temporais a partir da Silver
#    Uso típico (no final do pipeline, depois de gerar Silver + Gold):
#      duck = DuckDBManager()
#      duck.register_bronze_view("gastos-diretos")
#      duck.register_silver_view("gastos-diretos")
#      duck.register_gold_tables_from_parquet("gastos-diretos")
#      duck.create_temporal_tables("gastos-diretos")
#      duck.close()
class DuckDBManager:

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DUCKDB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger.info("Abrindo banco DuckDB em: %s", self.db_path)

        self.con = duckdb.connect(str(self.db_path))

    # Utilidades internas
    #  Retorna um pattern glob compatível com DuckDB, usando '/'.
    #  Ex.: dataset/silver/gastos-diretos/**/*.parquet
    def _parquet_glob(self, base: Path) -> str:
        return str(base / "**/*.parquet").replace("\\", "/")

    def _table_exists(self, name: str) -> bool:
        res = self.con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
            [name.lower()],
        ).fetchone()
        return res is not None

    def _view_exists(self, name: str) -> bool:
        res = self.con.execute(
            "SELECT 1 FROM information_schema.views WHERE table_name = ?",
            [name.lower()],
        ).fetchone()
        return res is not None

    # Retorna lista de colunas de uma tabela/view no DuckDB.
    def _get_columns(self, relation: str) -> List[str]:
        cur = self.con.execute(f"SELECT * FROM {relation} LIMIT 0")
        return [c[0] for c in cur.description]

    # Views Bronze / Silver
    def register_bronze_view(self, dataset_name: str = "gastos-diretos") -> None:
        sanitized = sanear_nome(dataset_name)
        bronze_path = BRONZE_DIR / dataset_name

        if not bronze_path.exists():
            logger.warning("Pasta Bronze não encontrada: %s", bronze_path)
            return

        pattern = self._parquet_glob(bronze_path)
        view_name = f"bronze_{sanitized}"

        logger.info("Criando view Bronze: %s → %s", view_name, pattern)
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT *
            FROM read_parquet('{pattern}');
            """
        )

    def register_silver_view(self, dataset_name: str = "gastos-diretos") -> None:
        sanitized = sanear_nome(dataset_name)
        silver_path = SILVER_DIR / dataset_name

        if not silver_path.exists():
            logger.warning("Pasta Silver não encontrada: %s", silver_path)
            return

        pattern = self._parquet_glob(silver_path)
        view_name = f"silver_{sanitized}"

        logger.info("Criando view Silver: %s → %s", view_name, pattern)
        self.con.execute(
            f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT *
            FROM read_parquet('{pattern}');
            """
        )

    # Carregar Parquets da Gold como tabelas DuckDB
    #  Lê todos os .parquet em dataset/gold/<dataset_name> e cria tabelas
    #  no DuckDB com o nome gold_<basename_sanitizado>.
    #   Ex.:
    #     dataset/gold/gastos-diretos/gastos_temporais.parquet
    #     → gold_gastos_temporais
    def register_gold_tables_from_parquet(self, dataset_name: str = "gastos-diretos") -> None:
        gold_path = GOLD_DIR / dataset_name
        if not gold_path.exists():
            logger.warning("Pasta Gold não encontrada: %s", gold_path)
            return

        arquivos = list(gold_path.glob("*.parquet"))
        if not arquivos:
            logger.warning("Nenhum arquivo .parquet encontrado em: %s", gold_path)
            return

        logger.info("Registrando %d tabelas Gold a partir de Parquet...", len(arquivos))

        for arq in arquivos:
            stem = arq.stem  # ex.: 'gastos_temporais'
            tabela = f"gold_{sanear_nome(stem)}"
            path_str = str(arq).replace("\\", "/")

            logger.info("Criando tabela %s a partir de %s", tabela, path_str)
            self.con.execute(
                f"""
                CREATE OR REPLACE TABLE {tabela} AS
                SELECT *
                FROM read_parquet('{path_str}');
                """
            )

        logger.info("Tabelas Gold (Parquet) registradas com sucesso.")

    # Tabelas de séries temporais ricas a partir da Silver
    # cria tabelas de séries temporais ricas a partir da view Silver:
    #  - gold_<dataset>_mensal
    #  - gold_<dataset>_orgao_mensal
    #  - gold_<dataset>_funcao_mensal
    #  - gold_<dataset>_programa_mensal
    #  - gold_<dataset>_grupo_despesa_mensal
    #  - gold_<dataset>_unidade_gestora_mensal
    def create_temporal_tables(self, dataset_name: str = "gastos-diretos") -> None:
        sanitized = sanear_nome(dataset_name)
        view_name = f"silver_{sanitized}"

        if not self._view_exists(view_name):
            logger.error(
                "View %s não encontrada. Execute register_silver_view primeiro.",
                view_name,
            )
            return

        cols = self._get_columns(view_name)
        logger.info("Colunas disponíveis em %s: %s", view_name, ", ".join(cols))

        required = {"ano", "mes", "valor"}
        missing = required - set(cols)
        if missing:
            logger.error(
                "View %s não possui colunas necessárias %s. Verifique a Silver.",
                view_name,
                missing,
            )
            return

        # --- Série temporal agregada geral (mensal) ---
        logger.info("Criando tabela gold_%s_mensal (série temporal geral)...", sanitized)
        self.con.execute(
            f"""
            CREATE OR REPLACE TABLE gold_{sanitized}_mensal AS
            WITH base AS (
                SELECT
                    ano,
                    mes,
                    (ano::VARCHAR || '-' || LPAD(mes::VARCHAR, 2, '0')) AS ano_mes,
                    SUM(valor) AS valor_total
                FROM {view_name}
                GROUP BY ano, mes
            )
            SELECT
                ano,
                mes,
                ano_mes,
                valor_total,
                LAG(valor_total) OVER (ORDER BY ano, mes) AS valor_mes_anterior,
                valor_total - LAG(valor_total) OVER (ORDER BY ano, mes) AS variacao_absoluta,
                CASE
                    WHEN LAG(valor_total) OVER (ORDER BY ano, mes) IS NULL
                         OR LAG(valor_total) OVER (ORDER BY ano, mes) = 0
                    THEN NULL
                    ELSE ROUND(
                        100.0 * (valor_total - LAG(valor_total) OVER (ORDER BY ano, mes))
                        / LAG(valor_total) OVER (ORDER BY ano, mes),
                        2
                    )
                END AS variacao_percentual,
                AVG(valor_total) OVER (
                    ORDER BY ano, mes
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS media_movel_3m,
                AVG(valor_total) OVER (
                    ORDER BY ano, mes
                    ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                ) AS media_movel_6m
            FROM base
            ORDER BY ano, mes;
            """
        )

        # Helper genérico pra criar tabelas por dimensão
        def criar_temporal_por_dim(col_dim: str, suffix: str) -> None:
            if col_dim not in cols:
                logger.info(
                    "Coluna %s não existe em %s. Tabela por %s não será criada.",
                    col_dim,
                    view_name,
                    suffix,
                )
                return

            tabela = f"gold_{sanitized}_{suffix}_mensal"
            logger.info(
                "Criando tabela %s (série temporal mensal por %s)...",
                tabela,
                col_dim,
            )

            self.con.execute(
                f"""
                CREATE OR REPLACE TABLE {tabela} AS
                WITH base AS (
                    SELECT
                        {col_dim} AS dim,
                        ano,
                        mes,
                        (ano::VARCHAR || '-' || LPAD(mes::VARCHAR, 2, '0')) AS ano_mes,
                        SUM(valor) AS valor_total,
                        COUNT(*)   AS qtd_pagamentos
                    FROM {view_name}
                    GROUP BY {col_dim}, ano, mes
                )
                SELECT
                    dim AS {col_dim},
                    ano,
                    mes,
                    ano_mes,
                    valor_total,
                    qtd_pagamentos,
                    LAG(valor_total) OVER (
                        PARTITION BY dim ORDER BY ano, mes
                    ) AS valor_mes_anterior,
                    valor_total - LAG(valor_total) OVER (
                        PARTITION BY dim ORDER BY ano, mes
                    ) AS variacao_absoluta,
                    CASE
                        WHEN LAG(valor_total) OVER (
                            PARTITION BY dim ORDER BY ano, mes
                        ) IS NULL
                             OR LAG(valor_total) OVER (
                                PARTITION BY dim ORDER BY ano, mes
                             ) = 0
                        THEN NULL
                        ELSE ROUND(
                            100.0 * (valor_total - LAG(valor_total) OVER (
                                PARTITION BY dim ORDER BY ano, mes
                            ))
                            / LAG(valor_total) OVER (
                                PARTITION BY dim ORDER BY ano, mes
                            ),
                            2
                        )
                    END AS variacao_percentual,
                    AVG(valor_total) OVER (
                        PARTITION BY dim
                        ORDER BY ano, mes
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                    ) AS media_movel_3m,
                    AVG(valor_total) OVER (
                        PARTITION BY dim
                        ORDER BY ano, mes
                        ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
                    ) AS media_movel_6m
                FROM base
                ORDER BY dim, ano, mes;
                """
            )

        # Cria tabelas de série temporal por várias dimensões (se existirem)
        criar_temporal_por_dim("nome_orgao", "orgao")
        criar_temporal_por_dim("nome_funcao", "funcao")
        criar_temporal_por_dim("nome_programa", "programa")
        criar_temporal_por_dim("nome_grupo_despesa", "grupo_despesa")
        criar_temporal_por_dim("nome_unidade_gestora", "unidade_gestora")
        criar_temporal_por_dim("nome_subfuncao", "subfuncao")

        logger.info("Tabelas temporais criadas com sucesso para %s.", dataset_name)

    # Utilitários diversos
    # Loga a lista de tabelas e views do banco.
    # Utilizado para debug rápido.
    def show_tables(self) -> None:
        logger.info("Listando tabelas e views do DuckDB:")
        tables = self.con.execute(
            "SELECT table_name, table_type FROM information_schema.tables ORDER BY table_type, table_name"
        ).fetchall()
        for name, ttype in tables:
            logger.info(" • %-8s %s", ttype, name)

    def close(self) -> None:
        logger.info("Fechando conexão com DuckDB.")
        self.con.close()
