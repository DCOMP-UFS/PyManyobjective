# Bem-vindo ao PyManyObjective

O **PyManyObjective** é um framework extensível desenvolvido na linguagem Python focado em resolver Problemas de Otimização com Muitos Objetivos (MaOPs - Many-Objective Optimization Problems). 

## Configuração do Ambiente

### Versão Recomendada do Python
- O projeto tem como alvo Python 3.9+. É recomendado o uso do Python 3.10 ou 3.11.
- É recomendado o uso do gerenciador de versões [pyenv](https://github.com/pyenv/pyenv).
- A versão atual do Python utilizada no projeto está armazenada no arquivo `.python-version` na raiz do repositório.

Instale e defina a versão do projeto:
```bash
pyenv install "$(cat .python-version)"
pyenv local "$(cat .python-version)"
python3 -V
```

### Ambiente Virtual Python
Crie um ambiente virtual Python na raiz do projeto:
```bash
python3 -m venv ./venv
```
Ative o ambiente:
```bash
source ./venv/bin/activate
```

### Instalação das Dependências
Instale as dependências usando o arquivo `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Como Navegar Pela Documentação

Nossa documentação foi reestruturada para facilitar seu entendimento e separar as seções essenciais:

1. [**Arquitetura do Framework:**](arquitetura.md) Entenda como as classes se relacionam através de um diagrama esquemático, e veja como implementar seus próprios problemas ou algoritmos por meio de herança e de métodos abstratos.
2. [**Getting Started:**](getting_started.md) Siga um passo a passo prático com códigos Python ensinando como instanciar um problema, configurar o algoritmo e rodar a otimização.
3. [**Problemas, Operadores e Métricas:**](problemas_metricas.md) Consulte as definições e funções presentes no projeto de forma separada do uso central, tais como funções de Benchmark (ZDT, DTLZ), métricas (IGD, GD) e operadores evolutivos (Cruzamento, Mutação, Seleção).
4. [**Algoritmos:**](algoritmos_evolucionarios.md) Saiba os detalhes teóricos e conceituais das implementações como NSGA-II, NSGA-III, Modelos Surrogates e DVL.