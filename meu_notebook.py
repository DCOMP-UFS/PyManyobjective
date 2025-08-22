import matplotlib.pyplot as plt
from src.problems import DTLZ
from src.MOEAs import NSGAII
from run import get_mutation, get_crossover, get_selection, get_sparsity

def plot_grafico_2d(pontos, title):

  x = []
  y = []

  for p in pontos:
    x.append(p[0])
    y.append(p[1])

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)

  ax.scatter(x, y, c='#C52535')
  ax.set_xlabel("f2")
  ax.set_ylabel("f1")
  ax.set_title(title)

  plt.show()


def mostrar_grafico(numberOfObjectives, k, maxEvaluations, populationSize, offSpringPopulationSize):

  problem = DTLZ.DTLZ2(numberOfObjectives, k) # Se pá colocar k = 9 para termos o n = 10, já que 'n' está em função de k+m-1

  nsga2 = NSGAII.NSGAII(

    problem=problem,
    maxEvaluations = maxEvaluations,
    populationSize = populationSize,
    offSpringPopulationSize = offSpringPopulationSize,
    crossover=None,
    mutation=None,
    selection=None,
    sparsity=None
  )

  nsga2.mutation = get_mutation("Polynomial", "resources/args_samples/Polynomial_args.json")
  nsga2.crossover = get_crossover("SBX", "resources/args_samples/SBX_args.json")
  nsga2.selection = get_selection("Binary")
  nsga2.sparsity = get_sparsity("CrowdingDistance")

  #title = f"NSGA-II - DTLZ2 com m={numberOfObjectives}, n=10, a={maxEvaluations}, p={populationSize}, f={offSpringPopulationSize}"
  nsga2.execute()
  front = [a.objectives for a in nsga2.paretoFront.getFront(0)] 
  plot_grafico_2d(front, title=f"NSGA-II - DTLZ2 com m={numberOfObjectives}, n=10, a={maxEvaluations}, p={populationSize}, f={offSpringPopulationSize}")

  return True

# Config. 1 - 1000 avaliaçoes
mostrar_grafico(2, 1, 1000, 50, 10)
mostrar_grafico(2, 1, 1000, 100, 100)
mostrar_grafico(2, 1, 1000, 200, 250)

# Config. 2 - 10000 avaliaçoes
mostrar_grafico(2, 1, 10000, 50, 10)
mostrar_grafico(2, 1, 10000, 100, 100)
mostrar_grafico(2, 1, 10000, 200, 250)


# Config. 3 - 50000 avaliaçoes
mostrar_grafico(2, 1, 50000, 50, 10)
mostrar_grafico(2, 1, 50000, 100, 100)
mostrar_grafico(2, 1, 50000, 200, 250)



# 4 - Executar o NSGA-II, no problema DTLZ2 com
# m = 2 objetivos
# n = 10 variáveis de decisão. (Se precisa de k=1) (Se pá colocar k = 9 para termos o n = 10, já que 'n' está em função de k+m-1)

# qtd de vezes q vou fazer a avaliação e o tamanho da populaçao respectivamente:

# a = maxEvaluations - 1000, 10000 e 50000
# p = populationSize - 50, 100 e 200
# f = offSpringPopulationSize - 10, 100, 250


# Config 1 - maxEvaluations = 1000:

  # maxEvaluations = 1000, p = 50 e f = 10
  # maxEvaluations = 1000, p = 100 e f = 100
  # maxEvaluations = 1000, p = 200 e f = 250



# Config 2 - maxEvaluations = 10000:
 
  # maxEvaluations = 10000, p = 50 e f = 10
  # maxEvaluations = 10000, p = 100 e f = 100
  # maxEvaluations = 10000, p = 200 e f = 250



# Config 3 - maxEvaluations = 50000:
 
  # maxEvaluations = 50000, p = 50 e f = 10
  # maxEvaluations = 50000, p = 100 e f = 100
  # maxEvaluations = 50000, p = 200 e f = 250




# Ordem: solution → Problem → ParetoFront → Alorithm → DTLZ → NSGAII