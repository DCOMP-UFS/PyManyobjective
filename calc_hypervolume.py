import json
import numpy as np
import matplotlib.pyplot as plt
from deap.tools._hypervolume.pyhv import hypervolume

name = "M1_25_100_nsga2"

t = 25

hypervolumes = []

for i in range(t):
	with open("results/" + name + "/F" + str(i), 'r') as F_file:
		for line in F_file.readlines():
			F = json.loads(line)
			F = np.array(F)
			hypervolumes.append(hypervolume(np.copy(F), np.array([1.0, 1.0])))

bestHypervolume = 0
bestHypervolume_i = 0
for i in range(t):
	print(hypervolumes[i])
	if hypervolumes[i] > bestHypervolume:
		bestHypervolume = hypervolumes[i]
		bestHypervolume_i = i