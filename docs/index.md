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
