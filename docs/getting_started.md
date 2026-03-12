# Getting Started - Passo a Passo

Neste guia, você aprenderá a configurar, instanciar e rodar uma rotina completa de otimização utilizando o **PyManyObjective**. O fluxo padrão e fundamental do framework consiste em definir o seu problema de otimização, preencher a configuração dos operadores de variação, enviá-los de forma abstrata para a instância principal do algoritmo evolutivo, e mandar executar.

## Fluxo de Execução (Esquema Visual)

```text
1. Instanciar Problema ---> 2. Configurar Operadores ---> 3. Instanciar Algoritmo ---> 4. Executar ---> 5. Resultados (Métricas)
     (ex: DTLZ1)            (Cruzamento, Mutação)             (ex: NSGA-III)                              (Fronteira, IGD)
```

## Passo 1: Instanciar um Problema

O framework possui diversos problemas de Benchmark implementados dentro da pasta `src/problems`. A instanciação sempre obedece ao padrão definido pela classe abstrata.

```python
from src.problems.DTLZ import DTLZ1

# Parâmetros necessários para formar as variáveis do problema
numberOfObjectives = 3
k = 10 # Fator numérico adotado para os problemas DTLZ
numberOfDecisionVariables = k + numberOfObjectives - 1

# Instanciar a classe do problema DTLZ1
problem = DTLZ1(numberOfObjectives, k)
```

## Passo 2: Configurar Operadores Evolucionários

Os algoritmos precisam de ferramentas para promover a variação na população e sua posterior seleção. Separamos sua implementação na estrutura para garantir a intercambialidade.

```python
from src.MOEAs.crossovers.SBXCrossover import SBXCrossover
from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation
from src.BinaryTournament import BinaryTournament

# Definir Crossover (Cruzamento Binário Simulado)
crossoverProbability = 0.9
crossoverDistributionIndex = 20.0
crossover = SBXCrossover(crossoverDistributionIndex, crossoverProbability)

# Definir Mutação (Mutação Polinomial)
mutationProbability = 1.0 / problem.numberOfDecisionVariables
mutationDistributionIndex = 20.0
mutation = PolynomialMutation(mutationProbability, mutationDistributionIndex)

# Definir Operador de Seleção
selection = BinaryTournament()
```

## Passo 3: Configurar e Instanciar o Algoritmo

Tendo em mãos os objetos referentes ao problema e aos operadores, procedemos para injetá-los no algoritmo que fará o trabalho central. Como exemplo, instanciaremos o NSGA-III.

```python
from src.MOEAs.NSGAIII import NSGAIII

maxEvaluations = 400
numberOfDivisions = 12 # Parâmetro do NSGA-III para suas partições de hiperplano (pontos de referência)

algorithm = NSGAIII(
    problem=problem,
    maxEvaluations=maxEvaluations,
    crossover=crossover,
    mutation=mutation,
    selection=selection,
    numberOfDivisions=numberOfDivisions
)
```

## Passo 4: Rodar o Framework e Obter os Resultados

Executar é um processo contido em um único comando `execute()`. Durante este tempo, o framework aplicará todas as etapas pre-codificadas e vai depositar os resultados dentro de `algorithm.paretoFront`.

```python
# Iniciar a Otimização
print("Iniciando a execução do NSGA-III...")
algorithm.execute()
print("Execução finalizada!")

# Conseguir o melhor grupo de dominância encontrado (Fronteira 0)
front = algorithm.paretoFront.getFront(0)

# Imprimir as variáveis mapeadas do espaço de objetivo para as soluções ótimas encontradas:
for index, solution in enumerate(front[:5]):
    print(f"Solução {index + 1} Objetivos:", solution.objectives)
```

## Passo Adicional: Utilizar Métricas 

Você pode medir o quão boa a sua otimização foi em relação a uma fronteira de referência (arquivos teóricos ou arquivos base obtidos de testes) através de uma classe métrica, como o IGD (Inverted Generational Distance).

```python
from src.QualityIndicator import IGD

# Supondo que você leu/gerou os pontos ideais em `file_front_ideal`
igd = IGD(file_front_ideal)

# A extração das features da população encontrada
objetivos_alcançados = [s.objectives for s in front]

# Calculando a distância inversa geracional final
resultado_igd = igd.calculate(objetivos_alcançados)
print("Distância de convergência da fronteira (IGD):", resultado_igd)
```

## Trabalhando com Algoritmos de Surrogate e Modelagem Inversa

Para testar o framework principal acoplado às técnicas baseadas em meta-modelagem ou *surrogates*, como o algoritmo M6 ou SMO, o seu MOEA de base entra como parâmetro encapsulado para os frameworks criados em `src/frameworks`.

```python
from src.frameworks.M6 import M6
from src.MOEAs.MRGA import MRGA
from pymoo.performance_indicator.kktpm import KKTPM

# Configurar o MOEA isolado
moea_base = MRGA(
    problem=problem, 
    maxEvaluations=10000, 
    populationSize=100, 
    offspringSize=50, 
    crossover=crossover, 
    mutation=mutation, 
    selection=selection, 
    sparsity=None, 
    R=None
)

# Acoplar no framework de Surrogate (M6)
framework = M6(
    EMO=moea_base, 
    problem=problem, 
    surrogate=None, # Define-se automaticamente pela lógica do pacote M6 (Kriging, Random Forest, SVR...)
    num_samples=100, 
    SEmax=500, 
    num_ref_dirs=3, 
    kktpm=KKTPM()
)

# Chamar a rotina otimizada
populacao_surrogate = framework.run()
```

Com este formato, é possível testar livremente configurações únicas e parametrizar experimentações acadêmicas facilmente.