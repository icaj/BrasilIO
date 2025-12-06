import logging
import os
from dotenv import load_dotenv
from raw import Raw_Dataset
from bronze import Bronze_Dataset
from silver import Silver_Dataset
from gold import Gold_Dataset
from brasilio import BrasilIO
from integracao_duckdb import DuckDBManager

# ---------------------------
# Constantes de Configuração
# ---------------------------
API_BASE_URL   = "https://brasil.io/api"
DATASET_SLUG   = "gastos-diretos"
NOME_TABELA    = "gastos"
API_TOKEN      = ""

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Carrega variáveis de ambiente do arquivo .env
# usado para ocultar o token do código
# ------------------------------------------------------
load_dotenv()
API_TOKEN      = os.getenv("BRASIL_IO_API_TOKEN")
if not API_TOKEN:
    logging.error("Variável de ambiente 'BRASIL_IO_API_TOKEN' não definida. Crie um arquivo .env ou defina-a no ambiente.")
    raise EnvironmentError(
        "Variável de ambiente 'BRASIL_IO_API_TOKEN' não definida. "
        "Crie um arquivo .env ou defina-a no ambiente."
    )

# faz o download de todas as páginas especificadas na variável 'total_paginas'
def download_paginas() -> int:
    # Varre todas as páginas até esvaziar.
    pagina = 1
    
    # por definição de Tinoco, o total de paginas buscadas será de 1000
    # total_paginas: Optional[int] = None
    total_paginas = 1000

    # pasta raw
    raw = Raw_Dataset()

    # Objeto BrasilIO
    api = BrasilIO(token=API_TOKEN, url=API_BASE_URL, dataset_slug=DATASET_SLUG, tabela=NOME_TABELA)
    
    while True:
        # verifica se a pagina ja foi baixada
        if raw.existe_pagina_raw(pagina):
            logging.info(f"Página {pagina} já existe na pasta RAW")
        else:
            # faz o download da página
            dados = api.busca_pagina(pagina)

            # Estrutura típica do Brasil.IO (DRF): count/next/previous/results
            itens = dados.get("results")
            if itens is None:
                # fallback para chaves alternativas
                itens = dados.get("data", dados if isinstance(dados, list) else [])

            logging.info(f"[INFO] Salvando página {pagina} na pasta raw")
            raw.grava_json(pagina, dados)

            if not itens:
                break

            api.espera_delay(1)

        pagina += 1
        if total_paginas is not None and pagina > total_paginas:
            break     

    # retorna lista com conteúdo de cada página
    return pagina

# inicio do programa
def main():
    # RAW - Download
    print(f"[INFO] Coletando de {API_BASE_URL}/dataset/{DATASET_SLUG}/{NOME_TABELA}/data/")
    paginas = download_paginas()
    logging.info(f"[INFO] Paginas lidas na execução: {paginas}")

    # RAW -> BRONZE
    brz = Bronze_Dataset()
    total_arquivos = brz.transformar_raw_para_bronze()
    logging.info(f"[INFO] {total_arquivos} arquivos processados")

    # BRONZE -> SILVER
    sil = Silver_Dataset(brz=brz)
    sil.processar_bronze_para_silver()

    # SILVER -> GOLD
    gld = Gold_Dataset()
    gld.processar()

    # Integração DuckDB
    duck = DuckDBManager()
    duck.register_bronze_view(DATASET_SLUG)
    duck.register_silver_view(DATASET_SLUG)
#    duck.create_basic_gold_tables(DATASET_SLUG)
    duck.close()
        
if __name__ == "__main__":
    main()
