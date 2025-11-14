import os, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

from raw_loader import existe_pagina_raw, grava_json
from bronze_transformer import transformar_raw_para_bronze
from silver_transformer import processar_bronze_para_silver
from gold_transformer import processar_gold
# ------------------------------------------------------
# Carrega variáveis de ambiente do arquivo .env
# usado para ocultar o token do código
# ------------------------------------------------------
load_dotenv()
API_TOKEN      = os.getenv("BRASIL_IO_API_TOKEN")
if not API_TOKEN:
    raise EnvironmentError(
        "Variável de ambiente 'BRASIL_IO_API_TOKEN' não definida. "
        "Crie um arquivo .env ou defina-a no ambiente."
    )
  
# ---------------------------
# Constantes de Configuração
# ---------------------------
API_BASE_URL   = "https://brasil.io/api"
DATASET_SLUG   = "gastos-diretos"
NOME_TABELA    = "gastos"
TAMANHO_PAGINA = 1000 # a api define o tamanho máximo de cada página é de 10000 bytes, mas por uma questão de velocidde defini em 1000
TIMEOUT        = 30
DIR_SAIDA      = Path("./dataset")
DIR_RAW        = DIR_SAIDA / "raw"
DIR_BRONZE     = DIR_SAIDA / "bronze"
DIR_GOLD       = DIR_SAIDA / "gold"
DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# API padrão Brasil.IO atual
# doc: https://blog.brasil.io/2020/10/10/como-acessar-os-dados-do-brasil-io/
DATA_URL       = f"{API_BASE_URL}/dataset/{DATASET_SLUG}/{NOME_TABELA}/data/"
    
# aguarda um delay (para ser usado entre as tentativas com erro)
def espera_delay(tentativa: int):
    time.sleep(min(60, (2 ** tentativa) + 0.1 * tentativa))

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

# devolve um header de autenticação
def header_autenticacao() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if API_TOKEN:
        # Observação: o blog do Brasil IO menciona:
        # `Authentication: Token <token>`. Por compatibilidade, envio ambos.
        h["Authentication"] = f"Token {API_TOKEN}"
        h["Authorization"]  = f"Token {API_TOKEN}"
    return h

# faz download de uma página especificada
def busca_pagina(pagina: int) -> Dict[str, Any]:
    qs = {"page": pagina, "page_size": TAMANHO_PAGINA}
    url = f"{DATA_URL}?{urlencode(qs)}"
    tentativas = 0

    print(f'busca_pagina(pagina = {pagina}, TAMANHO_PAGINA = {TAMANHO_PAGINA}), url = {url}')

    while True:
        try:
            r = requests.get(url, headers=header_autenticacao(), timeout=TIMEOUT)
            if r.status_code in (429,) or r.status_code >= 500:
                espera_delay(tentativas); tentativas += 1
                if tentativas > 5: r.raise_for_status()
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if tentativas >= 5:
                raise
            espera_delay(tentativas); tentativas += 1

# faz o download de todas as páginas especificadas na variável 'total_paginas'
def download_paginas() -> int:
    # Varre todas as páginas até esvaziar.
    pagina = 1
    # lista com conteúdo de cada pagina
    out: List[Dict[str, Any]] = []
    # por definição de Tinoco, o total de paginas buscadas será de 1000
    # total_paginas: Optional[int] = None
    total_paginas = 1000

    while True:
        # verifica se a pagina ja foi baixada
        if existe_pagina_raw(pagina):
            print(f"Página {pagina} já existe na pasta RAW")
        else:
            # faz o download da página
            dados = busca_pagina(pagina)

            # Estrutura típica do Brasil.IO (DRF): count/next/previous/results
            itens = dados.get("results")
            if itens is None:
                # fallback para chaves alternativas
                itens = dados.get("data", dados if isinstance(dados, list) else [])

            print(f"[INFO] Salvando página {pagina} na pasta raw")
            grava_json(pagina, dados)

            if not itens:
                break

            out.extend(itens)
            espera_delay(1)

        pagina += 1
        if total_paginas is not None and pagina > total_paginas:
            break     

    # retorna lista com conteúdo de cada página
    return pagina

# inicio do programa
def main():
    print(f"[INFO] Coletando de {DATA_URL}")

    # faz download de todas as paginas
    paginas = download_paginas()
    
    print(f"[INFO] Paginas lidas na execução: {paginas}")

    total_arquivos = transformar_raw_para_bronze()

    print(f"[INFO] {total_arquivos} arquivos processados")
    
    processar_bronze_para_silver()
    
    processar_gold()
    
if __name__ == "__main__":
    main()
