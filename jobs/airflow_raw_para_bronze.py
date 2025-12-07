import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.bronze import Bronze_Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Lê todos os JSON da pasta dataset/raw e grava Parquet particionado
# por ano/mes em dataset/bronze/<dataset_name>.
# Retorna: número de arquivos processados.
def raw_para_bronze(dataset_name: str = "gastos-diretos") -> int:
    brz = Bronze_Dataset(data_set=dataset_name)
    total_arquivos = brz.transformar_raw_para_bronze()
    logger.info("[BRONZE] %d arquivos processados.", total_arquivos)
    return total_arquivos

def run():
    return raw_para_bronze()

if __name__ == "__main__":
    run()
