# Bem-vindo ao PyManyObjective

O **PyManyObjective** é um framework extensível desenvolvido na linguagem Python focado em resolver Problemas de Otimização com Muitos Objetivos (MaOPs - Many-Objective Optimization Problems). 

Ele oferece implementações dos principais Algoritmos Evolucionários Multiobjetivos (MOEAs) da literatura, bem como algoritmos baseados em *surrogates* e Modelagem Inversa.

A Otimização Multiobjetivo busca encontrar um conjunto de soluções (Fronteira de Pareto) que represente o melhor compromisso entre objetivos conflitantes. Devido à alta exigência computacional de alguns problemas do mundo real, este framework engloba métodos focados em aprendizado de máquina (Machine Learning) e modelos de substituição (*surrogates* e *inverse models*) para acelerar e otimizar a busca pelas soluções ótimas.

## Setup

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

## Arquitetura do Framework
A estrutura orientada a objetos do framework contempla:
- **Problem**: Representação abstrata dos problemas de otimização.
- **Solution**: Representação de uma solução em espaço real com suas variáveis de decisão e valores dos objetivos.
- **ParetoFront**: Representação da fronteira de Pareto com métodos relacionados.
- **Algorithm**: Algoritmos base (MOEAs e Surrogates).
- **Operadores**: Seleção (`Selection`), Cruzamento (`Crossover`), e Mutação (`Mutation`).

