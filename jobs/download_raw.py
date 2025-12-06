import logging
import os
from dotenv import load_dotenv
from raw import Raw_Dataset
from brasilio import BrasilIO

API_BASE_URL = "https://brasil.io/api"
DATASET_SLUG = "gastos-diretos"
NOME_TABELA = "gastos"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Coleta páginas da API Brasil.io e salva em dataset/raw.
# Retorna: quantidade de páginas lidas na execução.
def download_paginas(total_paginas: int | None = 1000) -> int:
    load_dotenv()
    api_token = os.getenv("BRASIL_IO_API_TOKEN")
    if not api_token:
        raise EnvironmentError(
            "Variável de ambiente 'BRASIL_IO_API_TOKEN' não definida."
        )

    raw = Raw_Dataset()
    api = BrasilIO(
        token=api_token,
        url=API_BASE_URL,
        dataset_slug=DATASET_SLUG,
        tabela=NOME_TABELA,
    )

    pagina = 1
    while True:
        # se já existir, pula (idempotência p/ Airflow)
        if raw.existe_pagina_raw(pagina):
            logger.info("Página %d já existe na pasta RAW, pulando...", pagina)
            pagina += 1
        else:
            dados = api.busca_pagina(pagina)
            itens = dados.get("results")
            if itens is None:
                itens = dados.get("data", dados if isinstance(dados, list) else [])

            logger.info("[RAW] Salvando página %d", pagina)
            raw.grava_json(pagina, dados)

            # se não tem mais itens, para
            if not itens:
                break

            api.espera_delay(1)
            pagina += 1

        if total_paginas is not None and pagina >= total_paginas:
            break

    logger.info("[RAW] Total de páginas processadas nesta execução: %d", pagina)
    return pagina

# Função pensada para o PythonOperator do Airflow.
def run():
    return download_paginas()

if __name__ == "__main__":
    run()
