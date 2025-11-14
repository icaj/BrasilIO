import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

class Raw_Dataset:
    
    DATASET_SLUG   = "gastos-diretos"
    NOME_TABELA    = "gastos"
    DIR_RAW        = "./dataset/raw"
    DATA_HORA      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 
    def __init__(Self, dir="./dataset/raw", slug="gastos-diretos", tabela="gastos"):
        Self.DATASET_SLUG = slug
        Self.NOME_TABELA = tabela
        Self.DIR_RAW = Path(dir)
    
    # verifica se página já existe na pasta RAW
    def existe_pagina_raw(Self, pagina: int) -> bool:

        try:
            arquivos = os.listdir(Self.DIR_RAW)
            for arquivo in arquivos:
                if arquivo.endswith(f"_p{pagina:05d}.json"):
                    return True
            return False
        except Exception:
            return False

    # grava na pasta raw o json recebido
    def grava_json(Self, pagina: int, dados: Dict[str, Any]) -> Path:
        arquivo = f"{Self.DATASET_SLUG}_{Self.NOME_TABELA}_p{pagina:05d}.json"
        p = Self.DIR_RAW / arquivo
        p.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
        return p

    # le json da pasta raw
    def le_json_raw(Self, nome_arquivo: str) -> Dict[str, Any]:
        """
        Lê um arquivo JSON da pasta raw
        
        Args:
            nome_arquivo: Nome do arquivo JSON
        
        Returns:
            Dict com os dados do JSON
        """
        caminho = Self.DIR_RAW / nome_arquivo
        
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        return dados

    def listdir(Self) -> list[str]:
        try:
            arquivos = os.listdir(Self.DIR_RAW)
            return arquivos
        except Exception:
            return None
        
    

