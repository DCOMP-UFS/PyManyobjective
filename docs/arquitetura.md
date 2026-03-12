# Arquitetura do Framework

O **PyManyObjective** foi desenhado com base em princípios de Programação Orientada a Objetos, garantindo alta coesão e baixo acoplamento. A arquitetura central permite que pesquisadores estendam facilmente as funcionalidades adicionando novos problemas, operadores ou algoritmos ao instanciar os métodos das classes abstratas.

## Diagrama de Classes Principal

O diagrama abaixo ilustra de forma estruturada as dependências e o fluxo arquitetônico entre as classes abstratas e as concretas implementadas no framework:

```text
+--------------------------------------------------------------------------------+
|                                                                                |
|                                [ Main / Script ]                               |
|                                        |                                       |
|                                        v                                       |
|      +------------------------------------------------------------------+      |
|      |                        <<Abstract>>                              |      |
|      |                         Algorithm                                |      |
|      |------------------------------------------------------------------|      |
|      | - problem: Problem                                               |      |
|      | - crossover: Crossover                                           |      |
|      | - mutation: Mutation                                             |      |
|      | - selection: Selection                                           |      |
|      | - sparsity: Sparsity                                             |      |
|      | - population: Set[Solution]                                      |      |
|      | - paretoFront: ParetoFront                                       |      |
|      |------------------------------------------------------------------|      |
|      | + initializePopulation()                                         |      |
|      | + evolute()                                                      |      |
|      | + execute() <<Abstract>>                                         |      |
|      +------------------------------------------------------------------+      |
|                              |            |           |           |            |
|                              v            v           v           v            |
|                        +---------+   +----------+ +---------+ +----------+     |
|                        | Problem |   | Crossover| |Mutation | |Selection |     |
|                        +---------+   +----------+ +---------+ +----------+     |
|                                                                                |
|                                                                                |
|                                                                                |
+--------------------------------------------------------------------------------+
```

## Classes Base e Métodos Abstratos

Para adicionar novos componentes ao framework de maneira limpa, o usuário deve herdar das classes abstratas base listadas abaixo e instanciar seus métodos obrigatórios.

### 1. `Problem` (src/problems/Problem.py)
Classe abstrata que representa um problema de otimização multiobjetivo.

**Atributos Principais:**
- `numberOfObjectives`: Quantidade de funções objetivo a serem consideradas.
- `numberOfDecisionVariables`: Quantidade de variáveis de decisão da solução.
- `decisionVariablesLimit`: Limites (inferior e superior) permitidos no espaço de busca.

**Métodos Abstratos a Implementar nas Subclasses (ex: DTLZ1):**
- `evaluate(self, solution: Solution)`: É obrigatoriamente implementado para calcular os valores das funções objetivo (aptidão) de uma determinada solução e os atribuir a ela.
- `evaluateConstraints(self, solution: Solution)`: Trata as restrições que devem ser respeitadas pelas soluções no problema.
- `generateParetoFront(self)`: Gera ou importa a fronteira de Pareto verdadeira teórica para permitir o posterior cálculo de métricas de acurácia.

### 2. `Crossover` (src/MOEAs/crossovers/Crossover.py)
Classe abstrata para o operador genético de cruzamento (recombinação).

**Métodos Abstratos a Implementar:**
- `crossover(self, solutions: list, lowerBound, upperBound)`: Recebe uma lista de soluções ascendentes (pais) selecionadas e deve retornar uma lista correspondente de soluções filhas com informações combinadas dentro dos limites permitidos.

### 3. `Mutation` (src/MOEAs/mutations/Mutation.py)
Classe abstrata para o operador genético de mutação.

**Métodos Abstratos a Implementar:**
- `mutate(self, individual, lowerBound, upperBound)`: Submete o indivíduo da população aos processos da mutação, baseada na probabilidade de mutação configurada.
- `checkBounds(self, value, lower, upper)`: Checa e corrige os limites de uma variável alterada para garantir que não ultrapasse os domínios do problema.

### 4. `Algorithm` (src/MOEAs/Algorithm.py)
Representa o esquema geral para a criação de MOEAs (como NSGA-II e MOEA/D). O framework usa delegação, armazenando os operadores e o problema em seus atributos, de forma a reaproveitá-los em qualquer modelo.

**Métodos Abstratos a Implementar:**
- `execute(self)`: Método principal de execução e iteração do algoritmo. O desenvolvedor deve instanciar esse método para conter o ciclo evolutivo principal focado naquele algoritmo (criação de offspring, evolução, avaliação e sobrevivência), rodando até que se alcance a condição de término (geralmente avaliações máximas).