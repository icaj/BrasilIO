import time
from typing import Dict, Any
from urllib.parse import urlencode
import logging

import requests

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API padrão Brasil.IO atual
# doc: https://blog.brasil.io/2020/10/10/como-acessar-os-dados-do-brasil-io/
class BrasilIO:
    
    api_token      = ''
    api_base_url   = ""
    dataset_slug   = ""
    nome_tabela    = ""
    data_url       = ""

    TAMANHO_PAGINA = 1000 # a api define o tamanho máximo de cada página é de 10000 bytes, mas por uma questão de velocidde defini em 1000
    TIMEOUT        = 30

    # ------------------------------------------------------
    # Carrega variáveis de ambiente do arquivo .env
    # usado para ocultar o token do código
    # ------------------------------------------------------
    def __init__(self, token, url="https://brasil.io/api", dataset_slug="gastos-diretos", tabela="gastos"):
        self.api_token = token
        self.api_base_url = url
        self.dataset_slug = dataset_slug
        self.nome_tabela = tabela
        self.data_url = f"{url}/dataset/{dataset_slug}/{tabela}/data/"

    # devolve um header de autenticação
    def header_autenticacao(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_token:
            # Observação: o blog do Brasil IO menciona:
            # `Authentication: Token <token>`. Por compatibilidade, envio ambos.
            h["Authentication"] = f"Token {self.api_token}"
            h["Authorization"]  = f"Token {self.api_token}"
        return h

    # faz download de uma página especificada
    def busca_pagina(self, pagina: int) -> Dict[str, Any]:
        qs = {"page": pagina, "page_size": self.TAMANHO_PAGINA}
        url = f"{self.data_url}?{urlencode(qs)}"
        tentativas = 0

        print(f'busca_pagina(pagina = {pagina}, TAMANHO_PAGINA = {self.TAMANHO_PAGINA}), url = {url}')

        while True:
            try:
                r = requests.get(url, headers=self.header_autenticacao(), timeout=self.TIMEOUT)
                if r.status_code in (429,) or r.status_code >= 500:
                    self.espera_delay(tentativas); tentativas += 1
                    if tentativas > 5: r.raise_for_status()
                    continue
                r.raise_for_status()
                return r.json()
            except Exception:
                if tentativas >= 5:
                    raise
                self.espera_delay(tentativas); tentativas += 1

    # aguarda um delay (para ser usado entre as tentativas com erro)
    def espera_delay(self, tentativa: int):
        time.sleep(min(60, (2 ** tentativa) + 0.1 * tentativa))
