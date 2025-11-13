import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

DATASET_SLUG   = "gastos-diretos"
NOME_TABELA    = "gastos"
DIR_SAIDA      = Path("./dataset")
DIR_RAW        = DIR_SAIDA / "raw"
DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# verifica se página já existe na pasta RAW
def existe_pagina_raw(pagina: int) -> bool:

    try:
        arquivos = os.listdir(DIR_RAW)
        for arquivo in arquivos:
            if arquivo.endswith(f"_p{pagina:05d}.json"):
                return True
        return False
    except Exception:
        return False

# grava na pasta raw o json recebido
def grava_json(pagina: int, dados: Dict[str, Any]) -> Path:
    arquivo = f"{DATASET_SLUG}_{NOME_TABELA}_p{pagina:05d}.json"
    p = DIR_RAW / arquivo
    p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return p

