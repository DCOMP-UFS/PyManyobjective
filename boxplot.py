import json
import numpy as np
from deap.tools._hypervolume.pyhv import hypervolume
import math
import matplotlib.pyplot as plt

def convert_algo_name(name):
    splitted_name = name.split("_")
    if splitted_name[0] == "M1":
        return "M1 " + splitted_name[1] + "% " + splitted_name[2]
    else:
        return splitted_name[0] + " " + splitted_name[1]

def convert_names(algos):
    converted_names = []
    for algo in algos:
        converted_names.append(convert_algo_name(algo))
    return converted_names

def generate_boxplot(algos, t):
    all_hypervolumes = []

    for algo in algos:
        hypervolumes = []

        for i in range(t):
            with open("results/" + algo + "/F" + str(i), 'r') as F_file:
                for line in F_file.readlines():
                    F = json.loads(line)
                    F = np.array(F)
                    hypervolumes.append(hypervolume(np.copy(F), np.array([1.0, 1.0])))

        all_hypervolumes.append(hypervolumes)

    # for i in range(len(algos)):
    # 	for j in range(i + 1, len(algos)):
    # 		x = all_hypervolumes[i]
    # 		y = all_hypervolumes[j]
    # 		print("{} vs {}: {}".format(algos[i], algos[j], wilcoxon(x, y)))

    fig, ax1 = plt.subplots()
    plt.boxplot(all_hypervolumes)
    ax1.set_xticklabels(convert_names(algos),
                    rotation=45, fontsize=10)
    plt.show()
	