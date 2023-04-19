from scipy.stats import wilcoxon
import json
import numpy as np
from deap.tools._hypervolume.pyhv import hypervolume
import math
import scikit_posthocs as sp
import matplotlib.pyplot as plt

def convert_algo_name(name):
    splitted_name = name.split("_")
    if splitted_name[0] == "M1":
        return "M1 " + splitted_name[1] + "% " + splitted_name[2]
    else:
        return "NSGA-II " + splitted_name[1]

def convert_names(algos):
    converted_names = []
    for algo in algos:
        converted_names.append(convert_algo_name(algo))
    return converted_names

def generate_wilcoxon_heatmap(algos, t):
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

    pc = sp.posthoc_wilcoxon(all_hypervolumes)
    pc.columns = convert_names(algos)
    pc.index = convert_names(algos)

    cmap = ['1', '#fb6a4a',  '#08306b',  '#4292c6', '#c6dbef']
    heatmap_args = {'cmap': cmap, 'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
    sp.sign_plot(pc, **heatmap_args)
    plt.show()