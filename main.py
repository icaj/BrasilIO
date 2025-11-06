import os, json, time, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode

import requests
import pandas as pd

# ---------------------------
# Constantes de Configuração
# ---------------------------
API_BASE_URL   = "https://brasil.io/api"
DATASET_SLUG   = "gastos-diretos"
NOME_TABELA    = "gastos"
API_TOKEN      = "76a8419d5dfdb77c13abe53d03d7382178a03cb2"
TAMANHO_PAGINA = 1000 # a api define o tamanho máximo de cada página é de 10000 bytes, mas por uma questão de velocidde defini em 1000
TIMEOUT        = 30
DIR_SAIDA      = Path("./dataset")
DIR_RAW        = DIR_SAIDA / "raw"
DIR_BRONZE     = DIR_SAIDA / "bronze"
DIR_GOLD       = DIR_SAIDA / "gold"
DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# API padrão Brasil.IO atual
# doc: https://blog.brasil.io/2020/10/10/como-acessar-os-dados-do-brasil-io/
DATA_URL = f"{API_BASE_URL}/dataset/{DATASET_SLUG}/{NOME_TABELA}/data/"

# devolve um header de autenticação
def header_autenticacao() -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if API_TOKEN:
        # Observação: o blog do Brasil IO menciona:
        # `Authentication: Token <token>`. Por compatibilidade, envio ambos.
        h["Authentication"] = f"Token {API_TOKEN}"
        h["Authorization"]  = f"Token {API_TOKEN}"
    return h

# aguarda um delay (para ser usado entre as tentativas com erro)
def espera_delay(tentativa: int):
    time.sleep(min(60, (2 ** tentativa) + 0.1 * tentativa))

# grava na pasta raw o json recebido
def grava_json(pagina: int, dados: Dict[str, Any]) -> Path:
    arquivo = f"{DATASET_SLUG}_{NOME_TABELA}_{DATA_HORA}_p{pagina:05d}.json"
    p = DIR_RAW / arquivo
    p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return p

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
def extrai_todas_paginas() -> List[Dict[str, Any]]:
    # Varre todas as páginas até esvaziar.
    pagina = 1
    # lista com conteúdo de cada pagina
    out: List[Dict[str, Any]] = []
    # por definição de Tinoco, o total de paginas buscadas será de 1000
    # total_paginas: Optional[int] = None
    total_paginas = 1000

    while True:
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

        # calcula páginas quando possível
        #if total_paginas is None:
        #    count = dados.get("count")
        #    if isinstance(count, int) and count >= 0:
        #        total_paginas = math.ceil(count / TAMANHO_PAGINA)

        pagina += 1
        if total_paginas is not None and pagina > total_paginas:
            break

    # retorna lista com conteúdo de cada página
    return out

# transforma uma list em um parquet
def transforma_para_df(linhas: List[Dict[str, Any]]) -> pd.DataFrame:
    if not linhas:
        return pd.DataFrame()
    df = pd.json_normalize(linhas, max_level=1)

    # Tenta padronizar datas comuns (se existirem)
    for col in df.columns:
        if any(k in col.lower() for k in ["data", "dt_", "dt-", "dt.", "date"]):
            try:
                # Tenta o formato ISO 8601 padrão (ex: 2024-11-05 ou 2024-11-05T00:00:00Z)
                df[col] = pd.to_datetime(
                                df[col],
                                format="%Y-%m-%d",
                                errors="coerce",   # converte inválidos em NaT
                                utc=True
                            )
            except Exception:
                # fallback mais genérico, mas sem warnings
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # partição por ano/mês com base em uma coluna de data, se existir
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if date_cols:
        base = date_cols[0]
        df["ano"] = df[base].dt.year
        df["mes"] = df[base].dt.month
        df["dia"] = df[base].dt.strftime("%Y-%m")
    else:
        # sem coluna de data clara: particiona por run
        df["dia"] = DATA_HORA[:7]  # YYYYMM

    return df

# salva o parquet na pasta bronze
def grava_parquets(df: pd.DataFrame, dataset_name: str = f"{DATASET_SLUG}_{NOME_TABELA}") -> List[Path]:
    if df.empty:
        return []
    paths: List[Path] = []
    for part in sorted(df["dia"].unique()):
        part_df = df[df["dia"] == part].copy()
        out_dir = DIR_BRONZE / dataset_name / f"dia={part}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_{DATA_HORA}.parquet"
        # não levar colunas auxiliares de particionamento
        part_df.drop(columns=["dt"], inplace=True, errors="ignore")
        part_df.to_parquet(out_path, index=False)
        paths.append(out_path)
    return paths

# inicio do programa
def main():
    print(f"[INFO] Coletando de {DATA_URL}")
    linhas = extrai_todas_paginas()
    print(f"[INFO] Registros obtidos na execução: {len(linhas)}")

    if not linhas:
        print("[INFO] Nada a transformar.")
        return

    df = transforma_para_df(linhas)
    arquivos = grava_parquets(df)
    print("[INFO] Arquivos gravados na pasta bronze:")
    for a in arquivos:
        print(" -", a)

if __name__ == "__main__":
    main()
