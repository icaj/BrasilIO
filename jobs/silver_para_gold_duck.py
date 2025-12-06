import logging
from pathlib import Path
from gold import Gold_Dataset
from integracao_duckdb import DuckDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lê Silver, gera agregações Gold em Parquet e atualiza o banco DuckDB
#  com:
#  - view Bronze
#  - view Silver
#  - tabelas Gold (Parquet)
#  - tabelas de séries temporais (DuckDB)
def silver_para_gold_duck(dataset_name: str = "gastos-diretos") -> None:
    # 1) SILVER -> GOLD (Parquet)
    gld = Gold_Dataset(dataset_name=dataset_name)
    gld.processar()
    logger.info("[GOLD] Agregações geradas para dataset %s", dataset_name)

    # 2) DuckDB: views + tabelas
    duck = DuckDBManager()

    duck.register_bronze_view(dataset_name)
    duck.register_silver_view(dataset_name)
    duck.register_gold_tables_from_parquet(dataset_name)
    duck.create_temporal_tables(dataset_name)

    duck.show_tables()
    duck.close()
    logger.info("[DUCKDB] Atualização concluída para dataset %s", dataset_name)

def run():
    silver_para_gold_duck()

if __name__ == "__main__":
    run()
