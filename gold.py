from pathlib import Path
from datetime import datetime
from typing import Dict

import json
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# Configurações de diretórios
BASE_DIR = Path("./dataset")
DIR_SILVER = BASE_DIR / "silver"
DIR_GOLD = BASE_DIR / "gold"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Gold_Dataset:
    """
    Camada Gold:
    - Lê dados já limpos da Silver (Parquets particionados).
    - Faz validação adicional e limpeza mínima.
    - Cria agregações básicas e avançadas (séries temporais, Pareto, etc.).
    - Salva resultados em Parquet + metadados em JSON.
    """

    def __init__(self, dataset_name: str = "gastos-diretos") -> None:
        self.dataset_name = dataset_name
        self.silver_path = DIR_SILVER / dataset_name
        self.gold_path = DIR_GOLD / dataset_name
        self.metadata: Dict = {
            "processamento_timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "registros_processados": 0,
            "periodo_dados": {},
            "qualidade_dados": {},
        }

    # -------------------------------------------------------------------------
    # Validação / limpeza
    # -------------------------------------------------------------------------
    def validar_e_limpar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validando e limpando dados (Gold)...")

        df_clean = df.copy()
        registros_originais = len(df_clean)

        # Remover registros com detalhamento bloqueado no favorecido
        if "nome_favorecido" in df_clean.columns:
            mask_bloqueado = df_clean["nome_favorecido"].str.contains(
                "detalhamento das informações bloqueado",
                case=False,
                na=False,
            )
            registros_bloqueados = int(mask_bloqueado.sum())
            df_clean = df_clean[~mask_bloqueado].copy()
        else:
            registros_bloqueados = 0

        # Remover valores nulos ou zerados em valor
        df_clean = df_clean[df_clean["valor"].notna()].copy()
        df_clean = df_clean[df_clean["valor"] > 0].copy()

        # Garantir tipos corretos
        df_clean["ano"] = pd.to_numeric(df_clean["ano"], errors="coerce").astype("int64")
        df_clean["mes"] = pd.to_numeric(df_clean["mes"], errors="coerce").astype("int64")
        df_clean["valor"] = pd.to_numeric(df_clean["valor"], errors="coerce").astype(
            "float64"
        )

        # Criar coluna ano_mes
        df_clean["ano_mes"] = (
            df_clean["ano"].astype(str) + "-" + df_clean["mes"].astype(str).str.zfill(2)
        )

        # Estatísticas de limpeza
        registros_removidos = registros_originais - len(df_clean)
        logger.info(
            "Registros removidos: %d (%.2f%%)",
            registros_removidos,
            (registros_removidos / registros_originais * 100)
            if registros_originais > 0
            else 0,
        )
        logger.info(" - Bloqueados: %d", registros_bloqueados)
        logger.info(" - Valores inválidos: %d", registros_removidos - registros_bloqueados)

        self.metadata["qualidade_dados"] = {
            "registros_originais": registros_originais,
            "registros_limpos": len(df_clean),
            "registros_bloqueados": int(registros_bloqueados),
            "registros_invalidos": int(registros_removidos - registros_bloqueados),
            "taxa_aproveitamento": (
                f"{len(df_clean) / registros_originais * 100:.2f}%"
                if registros_originais > 0
                else "0.00%"
            ),
        }

        return df_clean

    # -------------------------------------------------------------------------
    # Métricas de qualidade / período
    # -------------------------------------------------------------------------
    def calcular_metricas_qualidade(self, df: pd.DataFrame) -> Dict:
        logger.info("Calculando métricas de qualidade...")

        metricas = {
            "periodo": {
                "ano_inicio": int(df["ano"].min()),
                "ano_fim": int(df["ano"].max()),
                "meses_disponiveis": int(df["ano_mes"].nunique()),
                "mes_mais_recente": df["ano_mes"].max(),
            },
            "cobertura": {
                "orgaos_unicos": int(df["nome_orgao"].nunique())
                if "nome_orgao" in df.columns
                else 0,
                "favorecidos_unicos": int(df["nome_favorecido"].nunique())
                if "nome_favorecido" in df.columns
                else 0,
                "programas_unicos": int(df["nome_programa"].nunique())
                if "nome_programa" in df.columns
                else 0,
                "subfuncoes_unicas": int(df["nome_subfuncao"].nunique())
                if "nome_subfuncao" in df.columns
                else 0,
            },
            "estatisticas_valor": {
                "total_gasto": float(df["valor"].sum()),
                "media": float(df["valor"].mean()),
                "mediana": float(df["valor"].median()),
                "desvio_padrao": float(df["valor"].std()),
                "minimo": float(df["valor"].min()),
                "maximo": float(df["valor"].max()),
            },
        }

        self.metadata["periodo_dados"] = metricas["periodo"]
        return metricas

    # -------------------------------------------------------------------------
    # Agregações básicas
    # -------------------------------------------------------------------------
    def criar_agregacoes_basicas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        logger.info("Criando agregações básicas...")
        agregacoes: Dict[str, pd.DataFrame] = {}

        # 1. Gastos por órgão e mês (com variação)
        if "nome_orgao" in df.columns:
            gastos_orgao_mes = (
                df.groupby(["nome_orgao", "ano", "mes", "ano_mes"])
                .agg({"valor": ["sum", "count", "mean", "median"]})
                .reset_index()
            )
            gastos_orgao_mes.columns = [
                "nome_orgao",
                "ano",
                "mes",
                "ano_mes",
                "valor_total",
                "qtd_pagamentos",
                "valor_medio",
                "valor_mediano",
            ]
            gastos_orgao_mes = gastos_orgao_mes.sort_values(
                ["nome_orgao", "ano", "mes"]
            )

            gastos_orgao_mes["valor_mes_anterior"] = gastos_orgao_mes.groupby(
                "nome_orgao"
            )["valor_total"].shift(1)
            gastos_orgao_mes["variacao_percentual"] = (
                (gastos_orgao_mes["valor_total"] - gastos_orgao_mes["valor_mes_anterior"])
                / gastos_orgao_mes["valor_mes_anterior"]
                * 100
            )

            agregacoes["gastos_por_orgao_mes"] = gastos_orgao_mes

        # 2. Gastos anuais totais por órgão
        if "nome_orgao" in df.columns:
            gastos_orgao_ano = (
                df.groupby(["nome_orgao", "ano"])
                .agg({"valor": ["sum", "count", "mean"]})
                .reset_index()
            )
            gastos_orgao_ano.columns = [
                "nome_orgao",
                "ano",
                "valor_total",
                "qtd_pagamentos",
                "valor_medio",
            ]
            gastos_orgao_ano = gastos_orgao_ano.sort_values(
                ["ano", "valor_total"], ascending=[True, False]
            )
            agregacoes["gastos_por_orgao_ano"] = gastos_orgao_ano

        # 3. Gastos totais por ano e mês (visão temporal geral)
        gastos_temporais = (
            df.groupby(["ano", "mes", "ano_mes"])
            .agg(
                {
                    "valor": ["sum", "count", "mean", "median"],
                    "nome_orgao": "nunique"
                    if "nome_orgao" in df.columns
                    else (lambda x: 0),
                    "nome_favorecido": "nunique"
                    if "nome_favorecido" in df.columns
                    else (lambda x: 0),
                }
            )
            .reset_index()
        )
        gastos_temporais.columns = [
            "ano",
            "mes",
            "ano_mes",
            "valor_total",
            "qtd_pagamentos",
            "valor_medio",
            "valor_mediano",
            "qtd_orgaos",
            "qtd_favorecidos",
        ]
        gastos_temporais = gastos_temporais.sort_values(["ano", "mes"])
        agregacoes["gastos_temporais"] = gastos_temporais

        # 4. Gastos por favorecido
        if "nome_favorecido" in df.columns:
            gastos_favorecido = (
                df.groupby(["nome_favorecido"])
                .agg(
                    {
                        "valor": ["sum", "count", "mean"],
                        "nome_orgao": "nunique"
                        if "nome_orgao" in df.columns
                        else (lambda x: 0),
                        "ano_mes": "nunique",
                    }
                )
                .reset_index()
            )
            gastos_favorecido.columns = [
                "nome_favorecido",
                "valor_total",
                "qtd_pagamentos",
                "valor_medio",
                "qtd_orgaos",
                "qtd_meses",
            ]
            gastos_favorecido = gastos_favorecido.sort_values(
                "valor_total", ascending=False
            )
            gastos_favorecido["percentual_total"] = (
                gastos_favorecido["valor_total"]
                / gastos_favorecido["valor_total"].sum()
                * 100
            )
            gastos_favorecido["percentual_acumulado"] = gastos_favorecido[
                "percentual_total"
            ].cumsum()
            agregacoes["gastos_por_favorecido"] = gastos_favorecido

        # 5. Gastos por subfunção
        if "nome_subfuncao" in df.columns:
            gastos_subfuncao = (
                df.groupby(["nome_subfuncao", "ano"])
                .agg({"valor": ["sum", "count", "mean"]})
                .reset_index()
            )
            gastos_subfuncao.columns = [
                "nome_subfuncao",
                "ano",
                "valor_total",
                "qtd_pagamentos",
                "valor_medio",
            ]
            gastos_subfuncao = gastos_subfuncao.sort_values(
                ["ano", "valor_total"], ascending=[True, False]
            )
            agregacoes["gastos_por_subfuncao"] = gastos_subfuncao

        # 6. Gastos por elemento de despesa
        if "nome_elemento_despesa" in df.columns:
            gastos_elemento = (
                df.groupby(["nome_elemento_despesa", "ano"])
                .agg({"valor": ["sum", "count", "mean"]})
                .reset_index()
            )
            gastos_elemento.columns = [
                "nome_elemento_despesa",
                "ano",
                "valor_total",
                "qtd_pagamentos",
                "valor_medio",
            ]
            gastos_elemento = gastos_elemento.sort_values(
                ["ano", "valor_total"], ascending=[True, False]
            )
            agregacoes["gastos_por_elemento_despesa"] = gastos_elemento

        # 7. Gastos por função
        if "nome_funcao" in df.columns:
            gastos_funcao = (
                df.groupby(["nome_funcao", "ano"])
                .agg({"valor": ["sum", "count"]})
                .reset_index()
            )
            gastos_funcao.columns = [
                "nome_funcao",
                "ano",
                "valor_total",
                "qtd_pagamentos",
            ]
            gastos_funcao = gastos_funcao.sort_values(
                ["ano", "valor_total"], ascending=[True, False]
            )
            agregacoes["gastos_por_funcao"] = gastos_funcao

        # 8. Gastos por programa
        if "nome_programa" in df.columns:
            gastos_programa = (
                df.groupby(["nome_programa", "ano"])
                .agg({"valor": ["sum", "count"]})
                .reset_index()
            )
            gastos_programa.columns = [
                "nome_programa",
                "ano",
                "valor_total",
                "qtd_pagamentos",
            ]
            gastos_programa = gastos_programa.sort_values(
                ["ano", "valor_total"], ascending=[True, False]
            )
            agregacoes["gastos_por_programa"] = gastos_programa

        logger.info("%d agregações básicas criadas", len(agregacoes))
        return agregacoes

    # -------------------------------------------------------------------------
    # Agregações avançadas
    # -------------------------------------------------------------------------
    def criar_agregacoes_avancadas(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        logger.info("Criando agregações avançadas...")
        agregacoes: Dict[str, pd.DataFrame] = {}

        # 1. TOP 100 maiores pagamentos
        top_pagamentos = (
            df.nlargest(100, "valor")[
                [
                    "ano_mes",
                    "nome_orgao",
                    "nome_favorecido",
                    "nome_programa",
                    "nome_elemento_despesa",
                    "valor",
                ]
            ]
            .reset_index(drop=True)
        )
        top_pagamentos["ranking"] = range(1, len(top_pagamentos) + 1)
        agregacoes["top_100_pagamentos"] = top_pagamentos

        # 2. Pareto por favorecido
        total_gasto = df["valor"].sum()
        concentracao = (
            df.groupby("nome_favorecido")["valor"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        concentracao["percentual"] = (
            concentracao["valor"] / total_gasto * 100 if total_gasto > 0 else 0
        )
        concentracao["percentual_acumulado"] = concentracao["percentual"].cumsum()
        concentracao["ranking"] = range(1, len(concentracao) + 1)

        favorecidos_80 = len(concentracao[concentracao["percentual_acumulado"] <= 80])
        concentracao["grupo_pareto"] = concentracao["percentual_acumulado"].apply(
            lambda x: "80% (A)"
            if x <= 80
            else ("95% (B)" if x <= 95 else "100% (C)")
        )
        agregacoes["concentracao_favorecidos"] = concentracao
        logger.info(
            "Análise de Pareto: %d favorecidos (%.2f%%) representam 80%% dos gastos",
            favorecidos_80,
            (favorecidos_80 / len(concentracao) * 100)
            if len(concentracao) > 0
            else 0,
        )

        # 3. Estatísticas por órgão
        stats_orgao = (
            df.groupby("nome_orgao")
            .agg(
                {
                    "valor": [
                        "sum",
                        "count",
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max",
                    ],
                    "nome_favorecido": "nunique",
                    "nome_programa": "nunique",
                    "ano_mes": "nunique",
                }
            )
            .reset_index()
        )
        stats_orgao.columns = [
            "nome_orgao",
            "valor_total",
            "qtd_pagamentos",
            "valor_medio",
            "valor_mediano",
            "desvio_padrao",
            "valor_minimo",
            "valor_maximo",
            "qtd_favorecidos",
            "qtd_programas",
            "qtd_meses_ativos",
        ]
        stats_orgao = stats_orgao.sort_values("valor_total", ascending=False)
        stats_orgao["percentual_total"] = (
            stats_orgao["valor_total"] / stats_orgao["valor_total"].sum() * 100
        )
        agregacoes["estatisticas_por_orgao"] = stats_orgao

        # 4. Tendências mensais
        tendencias = df.groupby(["ano_mes"])["valor"].sum().reset_index()
        tendencias = tendencias.sort_values("ano_mes")
        tendencias["valor_mes_anterior"] = tendencias["valor"].shift(1)
        tendencias["variacao_absoluta"] = (
            tendencias["valor"] - tendencias["valor_mes_anterior"]
        )
        tendencias["variacao_percentual"] = (
            tendencias["variacao_absoluta"] / tendencias["valor_mes_anterior"] * 100
        )
        tendencias["media_movel_3m"] = tendencias["valor"].rolling(window=3).mean()
        tendencias["media_movel_6m"] = tendencias["valor"].rolling(window=6).mean()
        agregacoes["tendencias_mensais"] = tendencias

        # 5. Sazonalidade (média por mês do ano)
        sazonalidade = (
            df.groupby("mes")
            .agg({"valor": ["sum", "mean", "count"]})
            .reset_index()
        )
        sazonalidade.columns = ["mes", "valor_total", "valor_medio", "qtd_pagamentos"]
        sazonalidade["percentual_ano"] = (
            sazonalidade["valor_total"] / sazonalidade["valor_total"].sum() * 100
        )
        meses_nome = {
            1: "Janeiro",
            2: "Fevereiro",
            3: "Março",
            4: "Abril",
            5: "Maio",
            6: "Junho",
            7: "Julho",
            8: "Agosto",
            9: "Setembro",
            10: "Outubro",
            11: "Novembro",
            12: "Dezembro",
        }
        sazonalidade["nome_mes"] = sazonalidade["mes"].map(meses_nome)
        agregacoes["sazonalidade_mensal"] = sazonalidade

        # 6. Outliers (>3 desvios padrão)
        media_global = df["valor"].mean()
        std_global = df["valor"].std()
        outliers = df[df["valor"] > media_global + 3 * std_global].copy()
        outliers = outliers.sort_values("valor", ascending=False)
        outliers["desvios_padrao"] = (outliers["valor"] - media_global) / std_global
        agregacoes["outliers_pagamentos"] = outliers[
            [
                "ano_mes",
                "nome_orgao",
                "nome_favorecido",
                "nome_programa",
                "nome_elemento_despesa",
                "valor",
                "desvios_padrao",
            ]
        ].reset_index(drop=True)
        logger.info(
            "Identificados %d pagamentos outliers (>3 desvios padrão)", len(outliers)
        )

        # 7. Matriz órgão x favorecido
        matriz_relacionamento = (
            df.groupby(["nome_orgao", "nome_favorecido"])
            .agg({"valor": ["sum", "count"]})
            .reset_index()
        )
        matriz_relacionamento.columns = [
            "nome_orgao",
            "nome_favorecido",
            "valor_total",
            "qtd_pagamentos",
        ]
        matriz_relacionamento = matriz_relacionamento.sort_values(
            ["nome_orgao", "valor_total"], ascending=[True, False]
        )
        agregacoes["matriz_orgao_favorecido"] = matriz_relacionamento

        logger.info("%d agregações avançadas criadas", len(agregacoes))
        return agregacoes

    # -------------------------------------------------------------------------
    # Persistência
    # -------------------------------------------------------------------------
    def salvar_agregacoes(self, agregacoes: Dict[str, pd.DataFrame]) -> None:
        logger.info("Salvando agregações na camada Gold...")
        self.gold_path.mkdir(parents=True, exist_ok=True)

        for nome, df in agregacoes.items():
            caminho = self.gold_path / f"{nome}.parquet"
            logger.info(" Salvando %s (%d registros) → %s", nome, len(df), caminho)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, caminho, compression="snappy")

        # Metadados
        metadata_path = self.gold_path / "_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(" %d agregações salvas em %s", len(agregacoes), self.gold_path)
        logger.info(" Metadados salvos em %s", metadata_path)

    # -------------------------------------------------------------------------
    # Relatório executivo (log)
    # -------------------------------------------------------------------------
    def gerar_relatorio_executivo(self, df: pd.DataFrame, metricas: Dict) -> None:
        logger.info("\n" + "=" * 80)
        logger.info("RELATÓRIO EXECUTIVO - CAMADA GOLD")
        logger.info("=" * 80)

        logger.info("\n PERÍODO DOS DADOS:")
        logger.info(
            " • Intervalo: %s a %s",
            metricas["periodo"]["ano_inicio"],
            metricas["periodo"]["ano_fim"],
        )
        logger.info(
            " • Meses disponíveis: %s", metricas["periodo"]["meses_disponiveis"]
        )
        logger.info(
            " • Mês mais recente: %s", metricas["periodo"]["mes_mais_recente"]
        )

        logger.info("\n VALORES TOTAIS:")
        valor_total = metricas["estatisticas_valor"]["total_gasto"]
        logger.info(" • Total gasto: R$ %,.2f", valor_total)
        logger.info(
            " • Média por pagamento: R$ %,.2f",
            metricas["estatisticas_valor"]["media"],
        )
        logger.info(
            " • Mediana: R$ %,.2f", metricas["estatisticas_valor"]["mediana"]
        )
        logger.info(
            " • Maior pagamento: R$ %,.2f", metricas["estatisticas_valor"]["maximo"]
        )

        logger.info("\n COBERTURA:")
        logger.info(" • Órgãos: %d", metricas["cobertura"]["orgaos_unicos"])
        logger.info(" • Favorecidos: %d", metricas["cobertura"]["favorecidos_unicos"])
        logger.info(" • Programas: %d", metricas["cobertura"]["programas_unicos"])
        logger.info(" • Subfunções: %d", metricas["cobertura"]["subfuncoes_unicas"])

        logger.info("\n TOP 5 ÓRGÃOS POR GASTO TOTAL:")
        top_orgaos = df.groupby("nome_orgao")["valor"].sum().nlargest(5)
        for i, (orgao, valor) in enumerate(top_orgaos.items(), 1):
            percentual = (valor / valor_total * 100) if valor_total > 0 else 0
            logger.info(" %d. %s: R$ %,.2f (%.2f%%)", i, orgao, valor, percentual)

        logger.info("\n TOP 5 FAVORECIDOS POR GASTO TOTAL:")
        top_favorecidos = df.groupby("nome_favorecido")["valor"].sum().nlargest(5)
        for i, (favorecido, valor) in enumerate(top_favorecidos.items(), 1):
            percentual = (valor / valor_total * 100) if valor_total > 0 else 0
            nome_truncado = (
                favorecido[:50] + "..." if len(favorecido) > 50 else favorecido
            )
            logger.info(
                " %d. %s: R$ %,.2f (%.2f%%)",
                i,
                nome_truncado,
                valor,
                percentual,
            )

        logger.info("\n QUALIDADE DOS DADOS:")
        qd = self.metadata["qualidade_dados"]
        logger.info(" • Registros processados: %s", f"{qd['registros_limpos']:,}")
        logger.info(" • Registros bloqueados: %s", f"{qd['registros_bloqueados']:,}")
        logger.info(" • Taxa de aproveitamento: %s", qd["taxa_aproveitamento"])
        logger.info("\n" + "=" * 80)

    # -------------------------------------------------------------------------
    # Pipeline Silver -> Gold
    # -------------------------------------------------------------------------
    def processar(self) -> None:
        logger.info("=" * 80)
        logger.info("INICIANDO PROCESSAMENTO GOLD: %s", self.dataset_name)
        logger.info("=" * 80)

        if not self.silver_path.exists():
            logger.error(" Pasta Silver não encontrada: %s", self.silver_path)
            return

        logger.info("Lendo dados da camada Silver em: %s", self.silver_path)

        try:
            # Lê todo o dataset Silver (particionado) via PyArrow
            dataset = ds.dataset(self.silver_path, format="parquet", partitioning="hive")
            table = dataset.to_table()
            df = table.to_pandas()

            if df.empty:
                logger.error(" Nenhum dado encontrado na Silver.")
                return

            logger.info("✓ %d registros carregados da camada Silver", len(df))
            logger.info(
                "Colunas disponíveis na Silver: %s",
                ", ".join(sorted(df.columns)),
            )

            # Garantir que ano/mes existam (devem vir da Silver)
            if "ano" not in df.columns or "mes" not in df.columns:
                logger.warning(
                    "Colunas 'ano'/'mes' ausentes. Tentando extrair de 'data_pagamento'..."
                )
                df["data_pagamento_dt"] = pd.to_datetime(
                    df.get("data_pagamento"), errors="coerce"
                )
                if "ano" not in df.columns:
                    df["ano"] = df["data_pagamento_dt"].dt.year
                if "mes" not in df.columns:
                    df["mes"] = df["data_pagamento_dt"].dt.month

                df = df[df["ano"].notna() & df["mes"].notna()].copy()

            df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
            df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
            df = df[df["ano"].notna() & df["mes"].notna()].copy()

            logger.info(
                "✓ Período: %d-%02d a %d-%02d",
                int(df["ano"].min()),
                int(df["mes"].min()),
                int(df["ano"].max()),
                int(df["mes"].max()),
            )

            self.metadata["registros_processados"] = len(df)

        except Exception as e:
            logger.error(" Erro ao ler dados da Silver: %s", e)
            return

        # Validação / limpeza
        df_clean = self.validar_e_limpar_dados(df)

        # Métricas
        metricas = self.calcular_metricas_qualidade(df_clean)

        # Agregações
        agregacoes_basicas = self.criar_agregacoes_basicas(df_clean)
        agregacoes_avancadas = self.criar_agregacoes_avancadas(df_clean)
        todas_agregacoes = {**agregacoes_basicas, **agregacoes_avancadas}

        # Salvar + relatório
        self.salvar_agregacoes(todas_agregacoes)
        self.gerar_relatorio_executivo(df_clean, metricas)

        logger.info("\n PROCESSAMENTO GOLD CONCLUÍDO COM SUCESSO!")
        logger.info(" Resultados disponíveis em: %s", self.gold_path)
