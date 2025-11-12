import os, json, time, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests
import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv

from silver_transformer import processar_bronze_para_silver
from gold_transformer import processar_silver_para_gold

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
#API_TOKEN      = "76a8419d5dfdb77c13abe53d03d7382178a03cb2"
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

def le_json_raw(nome_arquivo: str) -> Dict[str, Any]:
    """
    Lê um arquivo JSON da pasta raw
    
    Args:
        nome_arquivo: Nome do arquivo JSON
    
    Returns:
        Dict com os dados do JSON
    """
    caminho = DIR_RAW / nome_arquivo
    
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    
    with open(caminho, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    return dados

def transforma_json_para_df(dados: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforma dados JSON em DataFrame
    
    Args:
        dados: Dicionário com os dados JSON
    
    Returns:
        DataFrame com os dados normalizados
    """
    # Extrai os itens (results ou data)
    itens = dados.get("results")
    if itens is None:
        itens = dados.get("data", dados if isinstance(dados, list) else [])
    
    if not itens:
        print("[AVISO] Não há dados para processar")
        return pd.DataFrame()
    
    # Normaliza o JSON em DataFrame
    df = pd.json_normalize(itens, max_level=1)
    
    # Converte a coluna data_pagamento para datetime
    if 'data_pagamento' in df.columns:
        df['data_pagamento'] = pd.to_datetime(df['data_pagamento'], errors='coerce')
    
    # Converte valor para float
    if 'valor' in df.columns:
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    
    # Garante que ano e mes existem e são inteiros
    if 'ano' in df.columns:
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
    
    if 'mes' in df.columns:
        df['mes'] = pd.to_numeric(df['mes'], errors='coerce').astype('Int64')
    
    print(f"[INFO] DataFrame criado com {len(df)} registros")
    return df


def grava_parquet_particionado(df: pd.DataFrame, nome_dataset: str = "gastos-diretos") -> List[Path]:
    """
    Grava DataFrame em arquivos Parquet particionados por ano e mês
    
    Args:
        df: DataFrame com os dados
        nome_dataset: Nome base do dataset
    
    Returns:
        Lista com os caminhos dos arquivos gerados
    """
    if df.empty:
        print("[AVISO] DataFrame vazio, nada a gravar")
        return []
    
    # Verifica se as colunas de partição existem
    if 'ano' not in df.columns or 'mes' not in df.columns:
        print("[ERRO] DataFrame não possui colunas 'ano' e 'mes' para particionamento")
        return []
    
    paths: List[Path] = []
    
    # Remove registros com ano ou mês nulos
    df_valido = df.dropna(subset=['ano', 'mes'])
    
    if len(df_valido) < len(df):
        print(f"[AVISO] {len(df) - len(df_valido)} registros removidos por falta de ano/mês")
    
    # Agrupa por ano e mês
    for (ano, mes), grupo_df in df_valido.groupby(['ano', 'mes']):
        # Formata mes com dois dígitos
        mes_fmt = f"{int(mes):02d}"
        
        # Cria o diretório da partição
        partition_dir = DIR_BRONZE / nome_dataset / f"ano={int(ano)}" / f"mes={mes_fmt}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo
        arquivo = partition_dir / f"{nome_dataset}_ano{int(ano)}_mes{mes_fmt}.parquet"
        
        # Remove as colunas de partição do DataFrame (opcional)
        # grupo_df = grupo_df.drop(columns=['ano', 'mes'], errors='ignore')
        
        # Grava o parquet
        grupo_df.to_parquet(arquivo, index=False, engine='pyarrow')
        
        print(f"[INFO] Gravado: {arquivo} ({len(grupo_df)} registros)")
        paths.append(arquivo)
    
    return paths


def processar_json_para_parquet(nome_arquivo: str, nome_dataset: str = "gastos-diretos") -> List[Path]:
    """
    Função principal que processa um arquivo JSON e gera os Parquets particionados
    
    Args:
        nome_arquivo: Nome do arquivo JSON na pasta raw
        nome_dataset: Nome do dataset
    
    Returns:
        Lista com os caminhos dos arquivos gerados
    """
    print(f"[INFO] Processando arquivo: {nome_arquivo}")
    
    # 1. Lê o JSON
    dados = le_json_raw(nome_arquivo)
    print(f"[INFO] Total de registros no JSON: {dados.get('count', 'N/A')}")
    
    # 2. Transforma em DataFrame
    df = transforma_json_para_df(dados)
    
    if df.empty:
        print("[INFO] Nenhum dado para processar")
        return []
    
    # 3. Grava os Parquets particionados
    arquivos = grava_parquet_particionado(df, nome_dataset)
    
    print(f"\n[SUCESSO] {len(arquivos)} arquivo(s) Parquet gerado(s)")
    return arquivos


####---------------------------------------------------------------#####
    
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

# aguarda um delay (para ser usado entre as tentativas com erro)
def espera_delay(tentativa: int):
    time.sleep(min(60, (2 ** tentativa) + 0.1 * tentativa))

# grava na pasta raw o json recebido
def grava_json(pagina: int, dados: Dict[str, Any]) -> Path:
    arquivo = f"{DATASET_SLUG}_{NOME_TABELA}_p{pagina:05d}.json"
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

        # calcula páginas quando possível
        #if total_paginas is None:
        #    count = dados.get("count")
        #    if isinstance(count, int) and count >= 0:
        #        total_paginas = math.ceil(count / TAMANHO_PAGINA)

        pagina += 1
        if total_paginas is not None and pagina > total_paginas:
            break
        

    # retorna lista com conteúdo de cada página
    return pagina

# transforma uma list em um parquet
def transforma_para_df(linhas: List[Dict[str, Any]]) -> pd.DataFrame:
    if not linhas:
        return pd.DataFrame()
    df = pd.json_normalize(linhas, max_level=1)

    # 1) Tentamos converter colunas candidatas a data para datetime (UTC), sem format fixo
    #    para evitar warnings e NaT generalizado.
    tem_coluna_data_valida = False
    for col in df.columns:
        col_low = col.lower()
        if any(k in col_low for k in ["data", "dt_", "dt-", "dt.", "date"]):
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            # Considera "coluna valida" se há pelo menos um valor nao-nulo pós conversao
            if s.notna().any():
                df[col] = s
                tem_coluna_data_valida = True

    # 2) Coluna de partiçao 'dia' (YYYY-MM)
    if tem_coluna_data_valida:
        # Escolhe a primeira coluna datetime válida
        base_col = next(
            (c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])),
            None
        )
        if base_col is not None:
            df["dia"] = df[base_col].dt.strftime("%Y-%m")
            # Fallback para linhas com NaT
            df["dia"] = df["dia"].fillna(now_utc().strftime("%Y-%m"))
        else:
            df["dia"] = now_utc().strftime("%Y-%m")
    else:
        df["dia"] = now_utc().strftime("%Y-%m")

    return df

# salva o parquet na pasta bronze
def grava_parquets(df: pd.DataFrame, dataset_name: str = f"{DATASET_SLUG}_{NOME_TABELA}") -> List[Path]:
    if df.empty:
        return []
    paths: List[Path] = []
    # Garante que não haja NaN na chave de partição
    for part in sorted(x for x in df["dia"].dropna().unique()):
        part_df = df[df["dia"] == part].copy()
        if part_df.empty:
            continue  # não escreve partições vazias
        out_dir = DIR_BRONZE / dataset_name / f"dia={part}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_{DATA_HORA}.parquet"
        # Se quiser remover a coluna 'dia' do arquivo físico, comente a linha abaixo
        # part_df = part_df.drop(columns=["dia"], errors="ignore")
        part_df.to_parquet(out_path, index=False)
        paths.append(out_path)
    return paths

def transformar_raw_para_bronze() -> int:
    try:
        arquivos = os.listdir(DIR_RAW)
        for arquivo in arquivos:
            arqs = processar_json_para_parquet(arquivo)
        return len(arqs)
    except Exception:
        return 0
    
# inicio do programa
def main():
    print(f"[INFO] Coletando de {DATA_URL}")

    # faz download de todas as paginas
    paginas = download_paginas()
    
    print(f"[INFO] Paginas lidas na execução: {paginas}")

#   total_arquivos = transformar_raw_para_bronze()

#   print(f"[INFO] {total_arquivos} arquivos processados")
    
#    processar_bronze_para_silver()
    processar_silver_para_gold()
    
if __name__ == "__main__":
    main()
