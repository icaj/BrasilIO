import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

class Raw_Dataset:
    
    dataset_slug   = ""
    nome_tabela    = ""
    dir_raw        = ""
    DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 
    def __init__(Self, dir="./dataset/raw", slug="gastos-diretos", tabela="gastos"):
        Self.dataset_slug = slug
        Self.nome_tabela = tabela
        Self.dir_raw = Path(dir)
    
    # verifica se página já existe na pasta RAW
    # pagina: número da página a ser verificada
    # retorna: True se a página existe, False caso contrário
    def existe_pagina_raw(Self, pagina: int) -> bool:

        try:
            arquivos = os.listdir(Self.dir_raw)
            for arquivo in arquivos:
                if arquivo.endswith(f"_p{pagina:05d}.json"):
                    return True
            return False
        except Exception:
            return False

    # grava na pasta raw o json recebido
    # pagina: número da página a ser gravada
    # dados: dict com os dados a serem gravados
    def grava_json(Self, pagina: int, dados: Dict[str, Any]) -> Path:
        arquivo = f"{Self.dataset_slug}_{Self.nome_tabela}_p{pagina:05d}.json"
        p = Self.dir_raw / arquivo
        p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        return p

    # le json da pasta raw
    # nome_arquivo: nome do arquivo json a ler
    # retorna: dict com os dados do json
    def le_json_raw(Self, nome_arquivo: str) -> Dict[str, Any]:

        caminho = Self.dir_raw / nome_arquivo
        
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        return dados

    # lista arquivos na pasta raw
    # retorna: lista de nomes de arquivos na pasta raw
    def listdir(Self) -> list[str]:
        try:
            arquivos = os.listdir(Self.dir_raw)
            return arquivos
        except Exception:
            return None
        
    

