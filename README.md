# BrasilIO

Trabalho do Curso de Ciência de Dados da Matéria Engenharia de Dados da CESAR SCHOOL

Aluno: <b>IVO CAETANO DE ANDRADE JUNIOR</b>
Turma: Banco de dados - Noite

Analise de Dados Brasil IO

Ferramenta de extração de dados de gastos do Governo Federal da página Brasil IO ("https://brasil.io/api")

Utiiza python e bibliotecas para engenharia de dados.

Extrai informações da base de dados 'gastos-diretos' armazenando em arquivos json na pasta dataset/raw.

Após download, transforma os arquivos json para parquet e os armazena na pasta dataset/bronze

Além disso, os arquivos parquet da camada Silver podem ser materializados em um banco DuckDB
para consultas analíticas rápidas diretamente em SQL.

### Bibliotecas usadas:

requests

pandas

pyarrow

pyarrow.dataset

python-dotenv

streamlit

duckdb (opcional)

### Descrição

Este script em Phyton é um exercício de Engenharia de Dados que utiliza o site Brasil.io (https://brasil.io/) para explorar o dataset gastos-diretos, banco com informações sobre gastos do Governo Federal.
O script importa a base e guarda na pasta dataset/raw no formato JSON. Essas informações são então processadas e convertidas para o formato parquet e gravadas na pasta dataset/bronze. 

### Executando

Para executar crie um arquivo com o nome .env na pasta raiz e adione:

BRASIL_IO_API_TOKEN="chave"

Onde "chave" será sua string da API gerada no site BrasilIO

### Usando DuckDB (opcional)

Instale o conector Python:

```
pip install duckdb
```

Ao finalizar a execução do pipeline (raw → bronze → silver → gold), o script principal
criará o arquivo `dataset/brasilio.duckdb` com a tabela `brasilio.gastos_diretos` e a
view `brasilio.v_gastos_diretos_mensal`. Você pode abrir o arquivo com o cliente
interativo do DuckDB e consultar os dados, por exemplo:

```
duckdb dataset/brasilio.duckdb
SELECT * FROM brasilio.v_gastos_diretos_mensal ORDER BY ano DESC, mes DESC LIMIT 10;
```
