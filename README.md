# Datathon Fase05

## Sistema de Previsão de Risco e Análise Educacional - ONG Passos Mágicos

Este projeto foi desenvolvido como parte do Datathon – Fase 05 da Pós-Graduação em Data Analytics (FIAP).
O objetivo é analisar a base de dados educacionais da ONG Passos Mágicos, identificar padrões de alta performance, prever precocemente o risco de piora acadêmica/evasão dos alunos e disponibilizar uma aplicação web interativa usando o Streamlit para a equipe pedagógica.

## Objetivo do Projeto

- Analisar o histórico de desempenho e o desenvolvimento psicossocial dos alunos (PEDE 2022 a 2024).
- Identificar a "Anatomia da Alta Performance", mapeando o que diferencia um aluno de base de um aluno de elite (Ametista/Topázio).
- Desenvolver um modelo de Machine Learning focado em alto Recall para detectar alunos em risco antes que o problema se reflita nas notas.
- Entregar uma ferramenta gerencial (Dashboard) que automatize a análise de turmas inteiras.

## Base de Dados

A base utilizada é composta pelos registros anuais dos alunos, contendo variáveis educacionais exclusivas da metodologia Passos Mágicos:

- **Métricas Acadêmicas:** IDA (Desempenho), Defasagem escolar, Notas de Matemática, Português e Inglês.
- **Métricas Comportamentais e Emocionais:** IEG (Engajamento), IPS (Psicossocial), IAA (Autoavaliação) e IPP (Psicopedagógico).
- **Métricas Globais:** INDE (Índice de Desenvolvimento Educacional).
- **Classificações:** Fase escolar e "Pedra" (Quartzo, Ágata, Ametista, Topázio).

Os dados permitem uma visão holística 360º, unindo o estado emocional do aluno com suas notas e atitudes em sala de aula.

## Modelagem e Análise Preditiva

Foi desenvolvido um modelo de **Machine Learning (XGBoost Classifier)** para identificar a probabilidade de um aluno entrar em risco educacional no ano seguinte. Destaques da modelagem:

- **Foco no Recall (86,4%):** Ajustamos a linha de corte (threshold) do modelo para 41% em vez dos 50% padrões. A decisão de negócio por trás disso é: *é preferível gerar um falso alerta para a equipe pedagógica verificar um aluno que está bem, do que o algoritmo silenciar e deixar um aluno em risco passar despercebido.*
- **Engenharia de Recursos:** Criação de variáveis de "Tempo de Casa" e tradutores automáticos de strings para lidar com bases despadronizadas.

Principais bibliotecas utilizadas:
- pandas
- numpy (< 2.0.0)
- scikit-learn
- xgboost
- plotly
- matplotlib

## Aplicação Web – Streamlit

A aplicação foi desenvolvida em Streamlit e é dividida em dois módulos principais:

- **Simulador Individual:** Permite à equipe inserir notas e comportamentos hipotéticos para prever instantaneamente o termômetro de risco de um aluno específico.
- **Dashboard & Upload:** Permite fazer o upload das planilhas brutas (Excel/CSV) do ano. O sistema limpa os dados, corrige erros de digitação e devolve uma "Lista de Prioridade" com os alunos de maior risco, acompanhada de gráficos interativos da distribuição da turma.

Deploy realizado com sucesso no Streamlit Cloud, utilizando:
- Python 3.10 / 3.11
- Streamlit >= 1.40.0

## Principais Conclusões Estratégicas

Nossa análise exploratória revelou insights vitais para o negócio:

- **A Anatomia da Alta Performance:** O suporte psicológico (IPS) fornecido pela ONG é excelente e igualitário para todos. O que realmente diferencia um aluno comum de um aluno da "Elite" (Topázio/Ametista) não é apenas a nota, mas um **Engajamento (IEG) brutalmente superior**.
- **O Efeito Fadiga:** Analisando a jornada do aluno, descobrimos que o engajamento (IEG) e a conquista de pedras de topo caem vertiginosamente do Ano 0 para o Ano 3 na instituição, devido ao aumento da dificuldade escolar.
- **Recomendação:** A criação de um "Programa de Padrinhos", onde veteranos de alta performance mentorem os alunos no seu 2º e 3º ano para blindar o engajamento na fase mais crítica de evasão.

---

# Executando o Projeto Localmente no VS Code

Como configurar o ambiente e executar a aplicação Streamlit localmente utilizando o Visual Studio Code.

- Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

**Python 3.10 ou superior**
https://www.python.org/downloads/

**Visual Studio Code**
https://code.visualstudio.com/

**Extensão Python no VS Code (Microsoft)**

---

- Para verificar a versão do Python:

bash
python --version
Clonar o Repositório
Abra o Terminal do VS Code (Ctrl + ) e execute:

Bash
git clone [https://github.com/VinnihBr/Datathon_Fase05.git](https://github.com/VinnihBr/Datathon_Fase05.git)
cd Datathon_Fase05
OU

Baixar os arquivos em sua máquina:

Pastas: modelo

Arquivos: app.py, requirements.txt

Colocar todos os arquivos dentro de uma pasta para facilitar dentro do VS Code.

## **Dentro do VS Code:**

---

Criar e Ativar Ambiente Virtual (Recomendado)

Windows

Bash
python -m venv venv
venv\Scripts\activate
macOS / Linux

Bash
python3 -m venv venv
source venv/bin/activate
Após ativar, o nome do ambiente (venv) deve aparecer no terminal.

Instalar as Dependências

Com o ambiente virtual ativo:

Bash
pip install --upgrade pip
pip install -r requirements.txt
Configurar o Python no VS Code

Pressione Ctrl + Shift + P

Selecione Python: Select Interpreter

**Escolha o interpretador do ambiente virtual:**

---

venv\Scripts\python.exe (Windows)

venv/bin/python (macOS/Linux)

Executar a Aplicação Streamlit

No terminal do VS Code:

Bash
streamlit run app.py
Após executar o comando:
O Streamlit iniciará um servidor local e o navegador abrirá automaticamente em: http://localhost:8501

E agora você já pode usar a aplicação.

**Link da Aplicação no Streamlit Cloud:(https://datathon-fase05.streamlit.app/)**

## **Autores**

---

Bruno de Andrade

João Pedro

Vinicius Dias

Pós-Graduação em Data Analytics – FIAP
