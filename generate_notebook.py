import nbformat as nbf

nb = nbf.v4.new_notebook()

markdown_1 = """# Comparação e Validação do DVL Framework (DTLZ1 - 10 Objetivos)

Este notebook apresenta os resultados limpos do experimento utilizando o `DVLFramework` para o problema **DTLZ1** com 10 objetivos.

## Respostas às Dúvidas

**Por que os resultados pareciam ruins?**
Os resultados não eram ruins! O valor do hipervolume (HV) que estava na casa dos 9 milhões ocorreu porque a métrica não estava sendo **normalizada**. O ponto de referência do hipervolume para este problema (segundo a Tabela 17 da dissertação do Artur) é `[5.0, 5.0, ..., 5.0]`. O volume máximo sem normalização é $5.0^{10} = 9.765.625$. Portanto, um HV de $9.765.480$ equivale a um valor normalizado de $0.999985$ (praticamente $1.0$, que é perfeito). O script `run_dvl.py` foi atualizado para já entregar esse valor normalizado entre $[0, 1]$, permitindo a comparação direta com o artigo.

**Como podemos melhorar?**
Como o HV já está atingindo $1.0$ a partir de 1000 avaliações, o DVL Framework já está encontrando a fronteira de forma excelente. Uma das formas de melhorar a estabilidade em poucas avaliações (ex: 250) é utilizar as variações que se saíram melhor no estudo do Artur, como o uso de modelos Multilayer Perceptron (MLP) ou RVEA no lugar do NSGA-III (como MOEA acoplado). 

**O que significa IGD Estimado e Real (Final)?**
* **IGD Estimado (Estimated IGD):** É o valor do IGD avaliado logo após o modelo de *Machine Learning* (inverso) estimar as soluções no espaço de decisão, **antes** do algoritmo evolutivo (NSGA-III) começar a busca. Ele reflete a pura capacidade preditiva do modelo matemático.
* **IGD Final (Real):** É o IGD obtido após a população estimada pelo modelo ML ser refinada pelo Algoritmo Evolutivo (MOEA) até esgotar o orçamento de avaliações. Reflete o resultado do *framework* completo.
"""

code_1 = """import pandas as pd
from pathlib import Path
from run_dvl import run_experiment
import warnings
warnings.filterwarnings('ignore')

PROBLEM_K = 10
CASES = [
    {"evaluations": 250, "sample_size": 50},
    {"evaluations": 500, "sample_size": 112},
    {"evaluations": 1000, "sample_size": 200},
    {"evaluations": 1500, "sample_size": 200},
    {"evaluations": 10000, "sample_size": 300},
]

# Executa os casos
summary_rows = []
for case in CASES:
    result = run_experiment(
        problem_name="DTLZ1",
        m=10,
        e=case["evaluations"],
        sample_size=case["sample_size"],
        model_name="linear",
        seed=42,
        problem_k=PROBLEM_K,
    )
    summary_rows.append({
        "Avaliações": result["evaluations"],
        "Tamanho da Amostra (DVL)": result["sample_size"],
        "IGD Estimado (DVL)": round(result["estimated_igd"], 4),
        "IGD Final": round(result["final_igd"], 4),
        "HV Final (Normalizado)": round(result["final_hv"], 4),
    })

df_results = pd.DataFrame(summary_rows)
df_results
"""

markdown_2 = """## Comparação com a Dissertação do Artur

Abaixo encontram-se os valores extraídos da dissertação do Artur (*Tabela 20, pág. 91 - Valores de Hipervolume para 10 objetivos*), focando nas abordagens para o problema **DTLZ1**. A comparação foca no Hipervolume normalizado:
"""

code_2 = """artur_data = {
    "Avaliações": [250, 500, 1000, 1500, 10000],
    "Artur: DVL Isolado (HV)": ["0.725", "0.736", "0.846", "0.823", "0.821"],
    "Artur: NSGA-III Isolado (HV)": ["0.000", "0.000", "0.613", "0.050", "0.924"],
    "Artur: DVL+NSGA-III (HV)": ["0.671", "0.757", "0.829", "0.884", "0.943"],
}
df_artur = pd.DataFrame(artur_data)

# Juntando os dados para comparação final
df_compare = df_results[["Avaliações", "HV Final (Normalizado)"]].rename(columns={"HV Final (Normalizado)": "Nosso DVL+NSGA-III (HV)"})
df_compare = df_compare.merge(df_artur, on="Avaliações")
df_compare
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2)
]

with open('DVLFramework.ipynb', 'w') as f:
    nbf.write(nb, f)
