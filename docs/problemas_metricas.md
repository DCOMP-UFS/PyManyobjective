# Problemas e Métricas

Para assegurar a confiabilidade e testar o desempenho dos algoritmos desenvolvidos, o PyManyObjective possui a implementação de problemas padronizados e métricas consagradas pela literatura.

## Benchmarks de Otimização

### Família ZDT
Proposta por Zitzler, Deb e Thiele (2000), essas funções possuem dois objetivos e testam diferentes desafios do espaço de busca:
- **ZDT1**: Fronteira convexa.
- **ZDT3**: Fronteira desconexa (discreta).
- **ZDT6**: Problema não uniforme, com densidade variável e fronteira não convexa.

### Família DTLZ
Proposta por Deb, Thiele, Laumanns e Zitzler (2002), essa suíte foi desenhada para a validação de algoritmos em problemas com muitos objetivos (MaOPs - *Many-Objective*):
- **DTLZ1**: Superfície linear hiperplana, apresenta grande dificuldade de convergência devido à grande quantidade de ótimos locais.
- **DTLZ2**: Fronteira esférica (côncava), testa a capacidade do algoritmo cobrir toda a superfície de Pareto uniformemente.
- **DTLZ3**: Altamente multimodal, ideal para testar escape de ótimos locais.
- **DTLZ4**: Dificulta a manutenção de um conjunto diverso devido à densidade distorcida de soluções.
- **DTLZ5 e DTLZ6**: Fronteira de Pareto degenerada (unidimensional ou de menor dimensão).
- **DTLZ7**: Fronteira de Pareto totalmente desconectada (composta por múltiplas regiões distintas).

## Métricas de Qualidade

A medição de performance em otimização multiobjetivo se baseia essencialmente na **convergência** (o quão perto as soluções estão do ótimo real) e na **diversidade** (quão bem espalhadas as soluções estão pela fronteira de Pareto).

- **IGD (Inverted Generational Distance):** Mede a distância média euclidiana entre a verdadeira Fronteira de Pareto e a fronteira obtida pelo algoritmo. Quanto menor o valor de IGD, mais próxima e mais bem distribuída é a fronteira encontrada.
- **Hypervolume (HV):** Calcula o hipervolume (volume multi-dimensional) da região dominada pelo conjunto de soluções obtidas em relação a um ponto de referência (Nadir point). Valores maiores de Hypervolume indicam melhor performance em convergência e diversidade de soluções simultaneamente.