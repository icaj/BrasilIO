from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple
import numpy as np

# Configurações
BASE_DIR = Path("./dataset")
DIR_SILVER = BASE_DIR / "silver"
DIR_GOLD = BASE_DIR / "gold"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldTransformer:
    """
    Classe responsável por transformar dados da camada Silver em agregações
    e análises da camada Gold, processando múltiplos arquivos particionados.
    """
    
    def __init__(self, dataset_name: str = "gastos-diretos"):
        self.dataset_name = dataset_name
        self.silver_path = DIR_SILVER / dataset_name
        self.gold_path = DIR_GOLD / dataset_name
        self.metadata = {
            "processamento_timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "registros_processados": 0,
            "periodo_dados": {},
            "qualidade_dados": {}
        }
    
    def validar_e_limpar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida e limpa os dados, tratando valores bloqueados e inconsistências.
        """
        logger.info("Validando e limpando dados...")
        
        df_clean = df.copy()
        registros_originais = len(df_clean)
        
        # Remover registros com detalhamento bloqueado no favorecido
        mask_bloqueado = df_clean['nome_favorecido'].str.contains(
            'detalhamento das informações bloqueado', 
            case=False, 
            na=False
        )
        registros_bloqueados = mask_bloqueado.sum()
        df_clean = df_clean[~mask_bloqueado].copy()
        
        # Remover valores nulos ou zerados
        df_clean = df_clean[df_clean['valor'].notna()].copy()
        df_clean = df_clean[df_clean['valor'] > 0].copy()
        
        # Garantir tipos corretos
        df_clean['ano'] = df_clean['ano'].astype('int64')
        df_clean['mes'] = df_clean['mes'].astype('int64')
        df_clean['valor'] = df_clean['valor'].astype('float64')
        
        # Criar coluna ano_mes para facilitar análises temporais
        df_clean['ano_mes'] = df_clean['ano'].astype(str) + '-' + df_clean['mes'].astype(str).str.zfill(2)
        
        # Registrar estatísticas de limpeza
        registros_removidos = registros_originais - len(df_clean)
        logger.info(f"Registros removidos: {registros_removidos} ({registros_removidos/registros_originais*100:.2f}%)")
        logger.info(f"  - Bloqueados: {registros_bloqueados}")
        logger.info(f"  - Valores inválidos: {registros_removidos - registros_bloqueados}")
        
        self.metadata["qualidade_dados"] = {
            "registros_originais": registros_originais,
            "registros_limpos": len(df_clean),
            "registros_bloqueados": int(registros_bloqueados),
            "registros_invalidos": int(registros_removidos - registros_bloqueados),
            "taxa_aproveitamento": f"{len(df_clean)/registros_originais*100:.2f}%"
        }
        
        return df_clean
    
    def calcular_metricas_qualidade(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas de qualidade dos dados."""
        logger.info("Calculando métricas de qualidade...")
        
        metricas = {
            "periodo": {
                "ano_inicio": int(df['ano'].min()),
                "ano_fim": int(df['ano'].max()),
                "meses_disponiveis": int(df['ano_mes'].nunique()),
                "mes_mais_recente": df['ano_mes'].max()
            },
            "cobertura": {
                "orgaos_unicos": int(df['nome_orgao'].nunique()),
                "favorecidos_unicos": int(df['nome_favorecido'].nunique()),
                "programas_unicos": int(df['nome_programa'].nunique()),
                "subfuncoes_unicas": int(df['nome_subfuncao'].nunique())
            },
            "estatisticas_valor": {
                "total_gasto": float(df['valor'].sum()),
                "media": float(df['valor'].mean()),
                "mediana": float(df['valor'].median()),
                "desvio_padrao": float(df['valor'].std()),
                "minimo": float(df['valor'].min()),
                "maximo": float(df['valor'].max())
            }
        }
        
        self.metadata["periodo_dados"] = metricas["periodo"]
        
        return metricas
    
    def criar_agregacoes_basicas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cria agregações básicas e fundamentais."""
        logger.info("Criando agregações básicas...")
        agregacoes = {}
        
        # 1. Gastos por órgão e mês (com variação)
        gastos_orgao_mes = (
            df.groupby(['nome_orgao', 'ano', 'mes', 'ano_mes'])
            .agg({
                'valor': ['sum', 'count', 'mean', 'median']
            })
            .reset_index()
        )
        gastos_orgao_mes.columns = ['nome_orgao', 'ano', 'mes', 'ano_mes', 'valor_total', 'qtd_pagamentos', 'valor_medio', 'valor_mediano']
        gastos_orgao_mes = gastos_orgao_mes.sort_values(['nome_orgao', 'ano', 'mes'])
        
        # Calcular variação mensal
        gastos_orgao_mes['valor_mes_anterior'] = gastos_orgao_mes.groupby('nome_orgao')['valor_total'].shift(1)
        gastos_orgao_mes['variacao_percentual'] = (
            (gastos_orgao_mes['valor_total'] - gastos_orgao_mes['valor_mes_anterior']) / 
            gastos_orgao_mes['valor_mes_anterior'] * 100
        )
        
        agregacoes['gastos_por_orgao_mes'] = gastos_orgao_mes
        
        # 2. Gastos anuais totais por órgão
        gastos_orgao_ano = (
            df.groupby(['nome_orgao', 'ano'])
            .agg({
                'valor': ['sum', 'count', 'mean']
            })
            .reset_index()
        )
        gastos_orgao_ano.columns = ['nome_orgao', 'ano', 'valor_total', 'qtd_pagamentos', 'valor_medio']
        gastos_orgao_ano = gastos_orgao_ano.sort_values(['ano', 'valor_total'], ascending=[True, False])
        
        agregacoes['gastos_por_orgao_ano'] = gastos_orgao_ano
        
        # 3. Gastos totais por ano e mês (visão temporal geral)
        gastos_temporais = (
            df.groupby(['ano', 'mes', 'ano_mes'])
            .agg({
                'valor': ['sum', 'count', 'mean', 'median'],
                'nome_orgao': 'nunique',
                'nome_favorecido': 'nunique'
            })
            .reset_index()
        )
        gastos_temporais.columns = ['ano', 'mes', 'ano_mes', 'valor_total', 'qtd_pagamentos', 'valor_medio', 'valor_mediano', 'qtd_orgaos', 'qtd_favorecidos']
        gastos_temporais = gastos_temporais.sort_values(['ano', 'mes'])
        
        agregacoes['gastos_temporais'] = gastos_temporais
        
        # 4. Gastos por favorecido (top beneficiários)
        gastos_favorecido = (
            df.groupby(['nome_favorecido'])
            .agg({
                'valor': ['sum', 'count', 'mean'],
                'nome_orgao': 'nunique',
                'ano_mes': 'nunique'
            })
            .reset_index()
        )
        gastos_favorecido.columns = ['nome_favorecido', 'valor_total', 'qtd_pagamentos', 'valor_medio', 'qtd_orgaos', 'qtd_meses']
        gastos_favorecido = gastos_favorecido.sort_values('valor_total', ascending=False)
        gastos_favorecido['percentual_total'] = (gastos_favorecido['valor_total'] / gastos_favorecido['valor_total'].sum() * 100)
        gastos_favorecido['percentual_acumulado'] = gastos_favorecido['percentual_total'].cumsum()
        
        agregacoes['gastos_por_favorecido'] = gastos_favorecido
        
        # 5. Gastos por subfunção
        gastos_subfuncao = (
            df.groupby(['nome_subfuncao', 'ano'])
            .agg({
                'valor': ['sum', 'count', 'mean']
            })
            .reset_index()
        )
        gastos_subfuncao.columns = ['nome_subfuncao', 'ano', 'valor_total', 'qtd_pagamentos', 'valor_medio']
        gastos_subfuncao = gastos_subfuncao.sort_values(['ano', 'valor_total'], ascending=[True, False])
        
        agregacoes['gastos_por_subfuncao'] = gastos_subfuncao
        
        # 6. Gastos por elemento de despesa
        gastos_elemento = (
            df.groupby(['nome_elemento_despesa', 'ano'])
            .agg({
                'valor': ['sum', 'count', 'mean']
            })
            .reset_index()
        )
        gastos_elemento.columns = ['nome_elemento_despesa', 'ano', 'valor_total', 'qtd_pagamentos', 'valor_medio']
        gastos_elemento = gastos_elemento.sort_values(['ano', 'valor_total'], ascending=[True, False])
        
        agregacoes['gastos_por_elemento_despesa'] = gastos_elemento
        
        # 7. Gastos por função
        gastos_funcao = (
            df.groupby(['nome_funcao', 'ano'])
            .agg({
                'valor': ['sum', 'count']
            })
            .reset_index()
        )
        gastos_funcao.columns = ['nome_funcao', 'ano', 'valor_total', 'qtd_pagamentos']
        gastos_funcao = gastos_funcao.sort_values(['ano', 'valor_total'], ascending=[True, False])
        
        agregacoes['gastos_por_funcao'] = gastos_funcao
        
        # 8. Gastos por programa
        gastos_programa = (
            df.groupby(['nome_programa', 'ano'])
            .agg({
                'valor': ['sum', 'count']
            })
            .reset_index()
        )
        gastos_programa.columns = ['nome_programa', 'ano', 'valor_total', 'qtd_pagamentos']
        gastos_programa = gastos_programa.sort_values(['ano', 'valor_total'], ascending=[True, False])
        
        agregacoes['gastos_por_programa'] = gastos_programa
        
        logger.info(f"{len(agregacoes)} agregações básicas criadas")
        return agregacoes
    
    def criar_agregacoes_avancadas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cria agregações avançadas e análises especiais."""
        logger.info("Criando agregações avançadas...")
        agregacoes = {}
        
        # 1. TOP 100 maiores pagamentos
        top_pagamentos = (
            df.nlargest(100, 'valor')
            [['ano_mes', 'nome_orgao', 'nome_favorecido', 'nome_programa', 'nome_elemento_despesa', 'valor']]
            .reset_index(drop=True)
        )
        top_pagamentos['ranking'] = range(1, len(top_pagamentos) + 1)
        agregacoes['top_100_pagamentos'] = top_pagamentos
        
        # 2. Análise de concentração (Curva de Pareto por favorecido)
        total_gasto = df['valor'].sum()
        concentracao = (
            df.groupby('nome_favorecido')['valor']
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        concentracao['percentual'] = (concentracao['valor'] / total_gasto * 100)
        concentracao['percentual_acumulado'] = concentracao['percentual'].cumsum()
        concentracao['ranking'] = range(1, len(concentracao) + 1)
        
        # Identificar quantos favorecidos representam 80% dos gastos (Regra 80/20)
        favorecidos_80 = len(concentracao[concentracao['percentual_acumulado'] <= 80])
        concentracao['grupo_pareto'] = concentracao['percentual_acumulado'].apply(
            lambda x: '80% (A)' if x <= 80 else ('95% (B)' if x <= 95 else '100% (C)')
        )
        
        agregacoes['concentracao_favorecidos'] = concentracao
        
        logger.info(f"Análise de Pareto: {favorecidos_80} favorecidos ({favorecidos_80/len(concentracao)*100:.2f}%) representam 80% dos gastos")
        
        # 3. Estatísticas por órgão (resumo executivo)
        stats_orgao = df.groupby('nome_orgao').agg({
            'valor': ['sum', 'count', 'mean', 'median', 'std', 'min', 'max'],
            'nome_favorecido': 'nunique',
            'nome_programa': 'nunique',
            'ano_mes': 'nunique'
        }).reset_index()
        
        stats_orgao.columns = [
            'nome_orgao', 'valor_total', 'qtd_pagamentos', 'valor_medio', 
            'valor_mediano', 'desvio_padrao', 'valor_minimo', 'valor_maximo',
            'qtd_favorecidos', 'qtd_programas', 'qtd_meses_ativos'
        ]
        stats_orgao = stats_orgao.sort_values('valor_total', ascending=False)
        stats_orgao['percentual_total'] = (stats_orgao['valor_total'] / stats_orgao['valor_total'].sum() * 100)
        
        agregacoes['estatisticas_por_orgao'] = stats_orgao
        
        # 4. Tendências mensais (crescimento/redução)
        tendencias = df.groupby(['ano_mes'])['valor'].sum().reset_index()
        tendencias = tendencias.sort_values('ano_mes')
        tendencias['valor_mes_anterior'] = tendencias['valor'].shift(1)
        tendencias['variacao_absoluta'] = tendencias['valor'] - tendencias['valor_mes_anterior']
        tendencias['variacao_percentual'] = (tendencias['variacao_absoluta'] / tendencias['valor_mes_anterior'] * 100)
        tendencias['media_movel_3m'] = tendencias['valor'].rolling(window=3).mean()
        tendencias['media_movel_6m'] = tendencias['valor'].rolling(window=6).mean()
        
        agregacoes['tendencias_mensais'] = tendencias
        
        # 5. Sazonalidade (média por mês do ano)
        sazonalidade = (
            df.groupby('mes')
            .agg({
                'valor': ['sum', 'mean', 'count']
            })
            .reset_index()
        )
        sazonalidade.columns = ['mes', 'valor_total', 'valor_medio', 'qtd_pagamentos']
        sazonalidade['percentual_ano'] = (sazonalidade['valor_total'] / sazonalidade['valor_total'].sum() * 100)
        
        meses_nome = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho',
                      7: 'Julho', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
        sazonalidade['nome_mes'] = sazonalidade['mes'].map(meses_nome)
        
        agregacoes['sazonalidade_mensal'] = sazonalidade
        
        # 6. Outliers (pagamentos atípicos por desvio padrão)
        media_global = df['valor'].mean()
        std_global = df['valor'].std()
        
        outliers = df[df['valor'] > media_global + 3 * std_global].copy()
        outliers = outliers.sort_values('valor', ascending=False)
        outliers['desvios_padrao'] = (outliers['valor'] - media_global) / std_global
        
        agregacoes['outliers_pagamentos'] = outliers[
            ['ano_mes', 'nome_orgao', 'nome_favorecido', 'nome_programa', 
             'nome_elemento_despesa', 'valor', 'desvios_padrao']
        ].reset_index(drop=True)
        
        logger.info(f"Identificados {len(outliers)} pagamentos outliers (>3 desvios padrão)")
        
        # 7. Relacionamento Órgão x Favorecido (matriz de relacionamento)
        matriz_relacionamento = (
            df.groupby(['nome_orgao', 'nome_favorecido'])
            .agg({
                'valor': ['sum', 'count']
            })
            .reset_index()
        )
        matriz_relacionamento.columns = ['nome_orgao', 'nome_favorecido', 'valor_total', 'qtd_pagamentos']
        matriz_relacionamento = matriz_relacionamento.sort_values(['nome_orgao', 'valor_total'], ascending=[True, False])
        
        agregacoes['matriz_orgao_favorecido'] = matriz_relacionamento
        
        logger.info(f"{len(agregacoes)} agregações avançadas criadas")
        return agregacoes
    
    def salvar_agregacoes(self, agregacoes: Dict[str, pd.DataFrame]):
        """Salva todas as agregações em formato Parquet."""
        logger.info("Salvando agregações na camada Gold...")
        
        self.gold_path.mkdir(parents=True, exist_ok=True)
        
        for nome, df in agregacoes.items():
            caminho = self.gold_path / f"{nome}.parquet"
            logger.info(f"  Salvando {nome} ({len(df)} registros) → {caminho}")
            
            table = pa.Table.from_pandas(df)
            pq.write_table(table, caminho, compression='snappy')
        
        # Salvar metadados
        metadata_path = self.gold_path / "_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f" {len(agregacoes)} agregações salvas em {self.gold_path}")
        logger.info(f" Metadados salvos em {metadata_path}")
    
    def gerar_relatorio_executivo(self, df: pd.DataFrame, metricas: Dict):
        """Gera um relatório executivo em texto."""
        logger.info("\n" + "="*80)
        logger.info("RELATÓRIO EXECUTIVO - CAMADA GOLD")
        logger.info("="*80)
        
        logger.info(f"\n PERÍODO DOS DADOS:")
        logger.info(f"  • Intervalo: {metricas['periodo']['ano_inicio']} a {metricas['periodo']['ano_fim']}")
        logger.info(f"  • Meses disponíveis: {metricas['periodo']['meses_disponiveis']}")
        logger.info(f"  • Mês mais recente: {metricas['periodo']['mes_mais_recente']}")
        
        logger.info(f"\n VALORES TOTAIS:")
        valor_total = metricas['estatisticas_valor']['total_gasto']
        logger.info(f"  • Total gasto: R$ {valor_total:,.2f}")
        logger.info(f"  • Média por pagamento: R$ {metricas['estatisticas_valor']['media']:,.2f}")
        logger.info(f"  • Mediana: R$ {metricas['estatisticas_valor']['mediana']:,.2f}")
        logger.info(f"  • Maior pagamento: R$ {metricas['estatisticas_valor']['maximo']:,.2f}")
        
        logger.info(f"\n COBERTURA:")
        logger.info(f"  • Órgãos: {metricas['cobertura']['orgaos_unicos']}")
        logger.info(f"  • Favorecidos: {metricas['cobertura']['favorecidos_unicos']}")
        logger.info(f"  • Programas: {metricas['cobertura']['programas_unicos']}")
        logger.info(f"  • Subfunções: {metricas['cobertura']['subfuncoes_unicas']}")
        
        logger.info(f"\n TOP 5 ÓRGÃOS POR GASTO TOTAL:")
        top_orgaos = df.groupby('nome_orgao')['valor'].sum().nlargest(5)
        for i, (orgao, valor) in enumerate(top_orgaos.items(), 1):
            percentual = (valor / valor_total * 100)
            logger.info(f"  {i}. {orgao}: R$ {valor:,.2f} ({percentual:.2f}%)")
        
        logger.info(f"\n TOP 5 FAVORECIDOS POR GASTO TOTAL:")
        top_favorecidos = df.groupby('nome_favorecido')['valor'].sum().nlargest(5)
        for i, (favorecido, valor) in enumerate(top_favorecidos.items(), 1):
            percentual = (valor / valor_total * 100)
            nome_truncado = favorecido[:50] + '...' if len(favorecido) > 50 else favorecido
            logger.info(f"  {i}. {nome_truncado}: R$ {valor:,.2f} ({percentual:.2f}%)")
        
        logger.info(f"\n QUALIDADE DOS DADOS:")
        qd = self.metadata['qualidade_dados']
        logger.info(f"  • Registros processados: {qd['registros_limpos']:,}")
        logger.info(f"  • Registros bloqueados: {qd['registros_bloqueados']:,}")
        logger.info(f"  • Taxa de aproveitamento: {qd['taxa_aproveitamento']}")
        
        logger.info("\n" + "="*80)
    
    def processar(self):
        """Pipeline completo de processamento Silver → Gold."""
        logger.info("="*80)
        logger.info(f"INICIANDO PROCESSAMENTO: {self.dataset_name}")
        logger.info("="*80)
        
        # Verificar se a pasta Silver existe
        if not self.silver_path.exists():
            logger.error(f" Pasta Silver não encontrada: {self.silver_path}")
            return
        
        # Ler todos os arquivos Parquet particionados
        logger.info(f"Lendo dados particionados de: {self.silver_path}")
        try:
            # Ler dados com particionamento personalizado (YYYY/MM)
            # Vamos ler todos os parquets recursivamente
            parquet_files = list(self.silver_path.rglob("*.parquet"))
            
            if not parquet_files:
                logger.error(f" Nenhum arquivo .parquet encontrado em {self.silver_path}")
                return
            
            logger.info(f"✓ {len(parquet_files)} arquivo(s) .parquet encontrado(s)")
            
            # Ler todos os arquivos e concatenar
            dfs = []
            for file_path in parquet_files:
                try:
                    # Extrair ano e mês do caminho (ex: /2024/01/part-0.parquet)
                    parts = file_path.parts
                    ano = None
                    mes = None
                    
                    # Procurar ano (4 dígitos) e mês (2 dígitos) no caminho
                    for i, part in enumerate(parts):
                        if part.isdigit() and len(part) == 4:
                            ano = int(part)
                            # O próximo pode ser o mês
                            if i + 1 < len(parts) and parts[i + 1].isdigit() and len(parts[i + 1]) <= 2:
                                mes = int(parts[i + 1])
                            break
                    
                    df_temp = pd.read_parquet(file_path)
                    
                    # Adicionar ano e mês se foram encontrados no caminho
                    if ano is not None:
                        df_temp['ano'] = ano
                    if mes is not None:
                        df_temp['mes'] = mes
                    
                    dfs.append(df_temp)
                    logger.info(f"  Lido: {file_path.relative_to(self.silver_path)} ({len(df_temp):,} registros)")
                    
                except Exception as e:
                    logger.warning(f"    Erro ao ler {file_path}: {e}")
                    continue
            
            if not dfs:
                logger.error(" Nenhum arquivo foi lido com sucesso")
                return
            
            # Concatenar todos os DataFrames
            df = pd.concat(dfs, ignore_index=True)
            
            # Verificar se ano e mes existem, senão tentar extrair da data_pagamento
            if 'ano' not in df.columns or 'mes' not in df.columns or df['ano'].isna().any() or df['mes'].isna().any():
                logger.warning("Alguns registros sem 'ano'/'mes'. Extraindo da data_pagamento...")
                
                # Converter data_pagamento para datetime
                df['data_pagamento_dt'] = pd.to_datetime(df['data_pagamento'], errors='coerce')
                
                # Preencher ano e mês faltantes
                if 'ano' not in df.columns or df['ano'].isna().any():
                    df['ano'] = df['ano'].fillna(df['data_pagamento_dt'].dt.year)
                if 'mes' not in df.columns or df['mes'].isna().any():
                    df['mes'] = df['mes'].fillna(df['data_pagamento_dt'].dt.month)
                
                # Remover registros sem data válida
                df = df[df['ano'].notna() & df['mes'].notna()].copy()
                
            # Garantir tipos corretos
            df['ano'] = df['ano'].astype('int64')
            df['mes'] = df['mes'].astype('int64')
            
            logger.info(f"✓ Total: {len(df):,} registros carregados da camada Silver")
            logger.info(f"✓ Período: {df['ano'].min()}-{df['mes'].min():02d} a {df['ano'].max()}-{df['mes'].max():02d}")
            
            self.metadata["registros_processados"] = len(df)
            self.metadata["arquivos_processados"] = len(parquet_files)
            
        except Exception as e:
            logger.error(f" Erro ao ler dados: {e}")
            return
        
        # Validar e limpar dados
        df_clean = self.validar_e_limpar_dados(df)
        
        # Calcular métricas de qualidade
        metricas = self.calcular_metricas_qualidade(df_clean)
        
        # Criar agregações
        agregacoes_basicas = self.criar_agregacoes_basicas(df_clean)
        agregacoes_avancadas = self.criar_agregacoes_avancadas(df_clean)
        
        # Combinar todas as agregações
        todas_agregacoes = {**agregacoes_basicas, **agregacoes_avancadas}
        
        # Salvar resultados
        self.salvar_agregacoes(todas_agregacoes)
        
        # Gerar relatório executivo
        self.gerar_relatorio_executivo(df_clean, metricas)
        
        logger.info("\n PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info(f" Resultados disponíveis em: {self.gold_path}")


def processar_gold():
    """Função principal para execução do script."""
    transformer = GoldTransformer(dataset_name="gastos-diretos")
    transformer.processar()


