import math
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from deap.tools._hypervolume.pyhv import hypervolume
from src.ParetoFront import ParetoFront
from src.Population import Population

plt.tight_layout()
plt.figure(figsize=[3.0, 3.0])

def plot_non_dominated(F, algo_name):
    population = Population(F.shape[1], 1)
    population.decisionVariables = np.zeros((F.shape[0], 1))
    population.objectives = F

    paretoFront = ParetoFront()
    fronts = paretoFront.fastNonDominatedSort(population)

    non_dominated_F = F[fronts == 0]

    plt.plot(non_dominated_F[:,0], non_dominated_F[:, 1], 'bo')
    plt.savefig("plot_non_dominated_" + algo_name + ".png")
    plt.clf()

algos = ["M1_25_100_nsga2_breast_classic", "M1_50_100_nsga2_breast_classic", "M1_75_100_nsga2_breast_classic",
    "M1_25_1000_nsga2_breast_classic", "M1_50_1000_nsga2_breast_classic", "M1_75_1000_nsga2_breast_classic",
    "M1_25_10000_nsga2_breast_classic", "M1_50_10000_nsga2_breast_classic", "M1_75_10000_nsga2_breast_classic"]

def insert_plot_images(writer):
    worksheet = writer.sheets["Sheet1"]
    worksheet.set_row(9, 220)
    column_number = 1
    for i in range(len(algos)):
        worksheet.insert_image(chr(65 + column_number) + str("10"), "plot_non_dominated_" + algos[i] + ".png")
        column_number += 1

t = 25

data = {'': ["precisão média", "tempo médio", "hipervolume médio", "desvio padrão médio da precisão", "desvio padrão médio do tempo"]}

for algo in algos:
    avg_precision = 0
    avg_time = 0
    avg_hypervolume = 0
    precisions = []
    times = []
    hypervolumes = []
    all_F = []

    for i in range(t):
        with open("results/" + algo + "/bestPrecision" + str(i), 'r') as precision_file:
            for line in precision_file.readlines():
                avg_precision += float(line)
                precisions.append(float(line))
        with open("results/" + algo + "/T" + str(i), 'r') as time_file:
            for line in time_file.readlines():
                avg_time += float(line)
                times.append(float(line))
        with open("results/" + algo + "/F" + str(i), 'r') as F_file:
            for line in F_file.readlines():
                F = json.loads(line)
                F = np.array(F)
                avg_hypervolume += hypervolume(np.copy(F), np.array([1.0, 1.0]))
                hypervolumes.append(hypervolume(np.copy(F), np.array([1.0, 1.0])))
        with open("results/" + algo + "/F" + str(i), 'r') as F_file:
            for line in F_file.readlines():
                F = json.loads(line)
                for f in F:
                    all_F.append(f)

    avg_precision /= t
    avg_time /= t
    avg_hypervolume /= t

    p_dp = 0
    t_dp = 0
    for i in range(t):
        p_dp += (precisions[i] - avg_precision) ** 2
        t_dp += (times[i] - avg_time) ** 2
    p_dp /= t
    t_dp /= t
    p_dp = math.sqrt(p_dp)
    t_dp = math.sqrt(t_dp)

    all_F = np.array(all_F)
    all_F = np.unique(all_F, axis=0)
    plt.plot(all_F[:,0], all_F[:,1], 'bo')
    plt.savefig("plot_all_" + algo + ".png")
    plt.clf()

    plot_non_dominated(all_F, algo)

    data[algo] = [avg_precision, avg_time, avg_hypervolume, p_dp, t_dp]

df = pd.DataFrame(data)
writer = pd.ExcelWriter("demo.xlsx", engine="xlsxwriter")
df.to_excel(writer, sheet_name="Sheet1", index=False)

insert_plot_images(writer)

for column in df:
    column_length = max(df[column].astype(str).map(len).max(), len(column))
    col_idx = df.columns.get_loc(column)
    writer.sheets['Sheet1'].set_column(col_idx, col_idx, 45)

writer.close()