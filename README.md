# BrasilIO

TRABALHO DO CURSO DE CIENCIA DE DADOS DA MATERIA ENGENHARIA DE DADADOS DA CESAR SCHOOL

Aluno: IVO CAETANO DE ANDRADE JUNIOR
Turma: Banco de dados - Noite

Analise de Dados Brasil IO

Ferramenta de extração de dados de gastos do Governo Federal da página Brasil IO ("https://brasil.io/api")

Utiiza python e bibliotecas para engenharia de dados.

Extrai informações da base de dados 'gastos-diretos' armazenando em arquivos json na pasta dataset/raw.

Após download, transforma os arquivos json para parquet e os armazena na pasta dataset/bronze

### Bibliotecas usadas:

requests

pandas

pyarrow

pyarrow.dataset

python-dotenv

