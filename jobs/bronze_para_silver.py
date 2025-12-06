import logging
from bronze import Bronze_Dataset
from silver import Silver_Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lê Parquets da Bronze (particionados) e grava Silver já limpa e
# padronizada (SILVER_SCHEMA) em dataset/silver/<dataset_name>.
def bronze_para_silver(dataset_name: str = "gastos-diretos") -> None:
    brz = Bronze_Dataset(data_set=dataset_name)
    sil = Silver_Dataset(brz=brz)
    sil.processar_bronze_para_silver(dataset_name=dataset_name)
    logger.info("[SILVER] Pipeline Bronze → Silver concluído para %s", dataset_name)

def run():
    bronze_para_silver()

if __name__ == "__main__":
    run()
