"""Integração opcional da camada Silver com DuckDB.

Este módulo permite materializar os Parquets da camada Silver em um
banco DuckDB local, criando uma tabela e uma visão de agregação
mensal para consultas rápidas.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DuckDBIntegration:
    """Carrega dados da camada Silver para um banco DuckDB local."""

    def __init__(
        self,
        silver_dir: Path | str = Path("./dataset/silver"),
        database_path: Path | str = Path("./dataset/brasilio.duckdb"),
        dataset_name: str = "gastos-diretos",
    ) -> None:
        self.silver_dir = Path(silver_dir)
        self.database_path = Path(database_path)
        self.dataset_name = dataset_name

    def _duckdb_module(self):
        """Retorna o módulo duckdb se estiver disponível."""
        spec = importlib.util.find_spec("duckdb")
        if spec is None:
            logger.warning(
                "Biblioteca 'duckdb' não encontrada. Instale-a com 'pip install duckdb' para habilitar o carregamento."
            )
            return None

        import duckdb  # type: ignore

        return duckdb

    def carregar_silver_para_duckdb(self) -> Optional[Path]:
        """Cria/atualiza tabela DuckDB com os dados da camada Silver."""
        duckdb_module = self._duckdb_module()
        if duckdb_module is None:
            return None

        dataset_path = self.silver_dir / self.dataset_name
        if not dataset_path.exists():
            logger.error("Caminho da camada Silver não encontrado: %s", dataset_path)
            return None

        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        table_name = self.dataset_name.replace("-", "_")
        parquet_glob = f"{dataset_path}/**/*.parquet"

        logger.info("Carregando dados da Silver em DuckDB (%s)", self.database_path)
        conn = duckdb_module.connect(database=str(self.database_path))

        conn.execute("CREATE SCHEMA IF NOT EXISTS brasilio")
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE brasilio.{table_name} AS
            SELECT *
            FROM read_parquet(?, hive_partitioning = true)
            """,
            [parquet_glob],
        )

        conn.execute(
            f"""
            CREATE OR REPLACE VIEW brasilio.v_{table_name}_mensal AS
            SELECT
                ano,
                mes,
                SUM(valor) AS valor_total,
                COUNT(*) AS total_pagamentos
            FROM brasilio.{table_name}
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
        )

        previa = conn.execute(
            f"SELECT * FROM brasilio.v_{table_name}_mensal ORDER BY ano DESC, mes DESC LIMIT 5"
        ).fetchdf()
        logger.info("Prévia de agregação mensal via DuckDB:\n%s", previa)

        conn.close()
        return self.database_path


if __name__ == "__main__":
    integracao = DuckDBIntegration()
    integracao.carregar_silver_para_duckdb()
