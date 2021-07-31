from frameworks.SMO import SMO

import numpy as np
import math

from NSGAII import NSGAII
from SBXCrossover import SBXCrossover
from BinaryTournament import BinaryTournament
from PolynomialMutation import PolynomialMutation
from Sparsity import Sparsity

import matplotlib.pyplot as plt
from pymoo.factory import get_performance_indicator

class CrowdingDistance(Sparsity):
    def compute(self, front):
        for i in range(len(front)):
            front[i].sparsity = 0
        n_objs = len(front[0].objectives)
        for i in range(n_objs):
            mp = sorted(range(len(front)), key=lambda k: front[k].objectives[i])
            fmin = front[mp[0]].objectives[i]
            fmax = front[mp[len(front) - 1]].objectives[i]

            front[mp[0]].sparsity = front[mp[len(front) - 1]].sparsity = math.inf
            for j in range(1, len(front) - 1):
                front[mp[j]].sparsity = front[mp[j]].sparsity + (front[mp[j + 1]].objectives[i] - front[mp[j - 1]].objectives[i]) / (fmax - fmin)

def run(numberOfDecisionVariables, problem, comp_file, print_igd, print_x, print_y, show_graph):
    assert(numberOfDecisionVariables == problem.numberOfDecisionVariables)

    crossover = SBXCrossover(distributionIndex=20, crossoverProbability=0.9)
    mutation = PolynomialMutation(mutationProbability=1 / numberOfDecisionVariables, distributionIndex=20)
    selection = BinaryTournament()

    crowdingDistance = CrowdingDistance()

    nsgaii = NSGAII(problem=problem,
               maxEvaluations=10000,
               populationSize=100,
               offSpringPopulationSize=100,
               crossover=crossover,
               mutation=mutation,
               selection=selection,
               sparsity=crowdingDistance)

    smo = SMO(problem=problem, sample_size=300, t_max=500, t_parada=10000, EMO=nsgaii)
    my_front = smo.run()

    true_front = []
    with open(comp_file, 'r') as zdt_file:
        for line in zdt_file.readlines():
            xy = [float(line.split()[0]), float(line.split()[1])]
            true_front.append(xy)

    F = []
    for solution in my_front:
        F.append(solution.objectives)

    if print_igd:
        igd_p = get_performance_indicator("igd", np.array(true_front))
        print(igd_p.calc(np.array(F)))
        print("")

    if print_x:
        for x in my_front:
            print(x)
        print("")

    if print_y:
        for y in F:
            print(y)
        print("")

    F_x = []
    F_y = []
    for xy in F:
        F_x.append(xy[0])
        F_y.append(xy[1])

    true_x = []
    true_y = []
    for xy in true_front:
        true_x.append(xy[0])
        true_y.append(xy[1])

    if show_graph:
        plt.plot(F_x, F_y, 'bo')
        plt.plot(true_x, true_y, 'ro', markersize=2)
        plt.axis([0, 1.0, 0, 1.2])
        plt.show()

    return igd_p.calc(np.array(F))

