# BrasilIO

graph TD
    Diretoria --> Gerente1[Gerente de Vendas]
    Diretoria --> Gerente2[Gerente de TI]
    Gerente1 --> Vendedor1[Vendedor A]
    Gerente1 --> Vendedor2[Vendedor B]
    Gerente2 --> Dev1[Desenvolvedor]
    
Trabalho do Curso de Ciência de Dados da Matéria Engenharia de Dados da CESAR SCHOOL

Aluno: <b>IVO CAETANO DE ANDRADE JUNIOR</b>
Turma: Banco de dados - Noite

Análise de Dados da Brasil IO

Ferramenta de extração de dados de gastos do Governo Federal da página Brasil IO ("https://brasil.io/api")

Utiiza python e bibliotecas para engenharia de dados.

Extrai informações da base de dados 'gastos-diretos' armazenando em arquivos json na pasta dataset/raw.

Após download, transforma os arquivos json para parquet e os armazena na pasta dataset/bronze

Em seguida faz limpeza e correção desses dados e armazena na pasta dataset/silver em parquet

Depois avalida esses dados a fim de extrair informações úteis e armazena pasta dataset/gold

Finalmente gera KPI importantes destas informações e os armazena em parquet otimizado e em base de daos duckdb

### Bibliotecas usadas:

requests

pandas

pyarrow

python-dotenv

duckdb

apache-airflow

### Descrição

Este script em Phyton é um exercício de Engenharia de Dados que utiliza o site Brasil.io (https://brasil.io/) para explorar o dataset gastos-diretos, banco com informações sobre gastos do Governo Federal.
O script faz todo o processo de ETL usando o modelo medalhao usando pastas raw, bronze, silver e gold.

### Executando

Esta aplicação funciona apenas em Linux.

Baixe os arquivos do repositório e vá até a pasta onde fez dawnload do repositório e execute:

bash instala_dependencias.sh

Isto instalará as dependencias necessárias do Python

Caso não possua instalado o Apache Airflow em seu comptuaodr, execute o comando abaixo:

bash instala_airflow.sh

Em seguida, basta executar na pasta do projeto o comando abaixo para iniciar:

bash start_airflow.sh

