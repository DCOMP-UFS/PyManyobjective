import json
import numpy as np
import matplotlib.pyplot as plt
from src.ParetoFront import ParetoFront
from src.Population import Population

name = "M1_25_100_nsga2"

t = 25

all_F = []

for i in range(t):
	with open("results/" + name + "/F" + str(i), 'r') as F_file:
		for line in F_file.readlines():
			F = json.loads(line)
			for f in F:
				all_F.append(f)

all_F = np.array(all_F)

population = Population(all_F.shape[1], 1)
population.decisionVariables = np.zeros((all_F.shape[0], 1))
population.objectives = all_F

paretoFront = ParetoFront()
fronts = paretoFront.fastNonDominatedSort(population)

non_dominated_all_F = all_F[fronts == 0]

plt.plot(all_F[:,0], all_F[:, 1], 'bo')
plt.savefig("plot_all.png")
plt.plot(non_dominated_all_F[:,0], non_dominated_all_F[:, 1], 'bo')
plt.savefig("plot_best.png")