import math
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from deap.tools._hypervolume.pyhv import hypervolume
from src.ParetoFront import ParetoFront
from src.Population import Population
from sklearn.metrics import roc_curve, auc
from run_hyper import get_problem

plt.tight_layout()
plt.figure(figsize=[3.0, 3.0])

def get_best_value_and_index(arr, maximize):
    best = arr[0]
    best_i = 0

    if maximize:
        for i in range(1, len(arr)):
            if arr[i] > best:
                best = arr[i]
                best_i = i
    else:
        for i in range(1, len(arr)):
            if arr[i] < best:
                best = arr[i]
                best_i = i

    return best, best_i

def plot_colored_all(F, algo_name, bestPrecision_i, bestAUC_i):
    population = Population(F.shape[1], 1)
    population.decisionVariables = np.zeros((F.shape[0], 1))
    population.objectives = F

    color_values = ["black", "green", "orange"]
    colors = np.zeros((F.shape[0]), dtype=np.int32)
    colors[bestPrecision_i] = 1
    colors[bestAUC_i] = 2
    colors = colors.tolist()

    for i in range(len(F)):
        plt.scatter(F[:,0][i], F[:,1][i], c=color_values[colors[i]])
    plt.savefig("plot_all_" + algo_name + ".png")
    plt.clf()

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

def plot_colored_non_dominated(F, algo_name, bestPrecision_i, bestAUC_i):
    population = Population(F.shape[1], 1)
    population.decisionVariables = np.zeros((F.shape[0], 1))
    population.objectives = F

    paretoFront = ParetoFront()
    fronts = paretoFront.fastNonDominatedSort(population)

    color_values = ["black", "green", "orange"]
    colors = np.zeros((F.shape[0]), dtype=np.int32)
    colors[bestPrecision_i] = 1
    colors[bestAUC_i] = 2
    colors[fronts == 0]
    colors = colors.tolist()

    non_dominated_F = F[fronts == 0]

    for i in range(len(non_dominated_F)):
        plt.scatter(non_dominated_F[:,0][i], non_dominated_F[:,1][i], c=color_values[colors[i]])
    plt.savefig("plot_non_dominated_" + algo_name + ".png")
    plt.clf()

def insert_plot_images(writer):
    worksheet = writer.sheets["Sheet1"]
    worksheet.set_row(11, 220)
    worksheet.set_row(12, 220)
    column_number = 1
    for i in range(len(algos)):
        worksheet.insert_image(chr(65 + column_number) + str("12"), "plot_all_" + algos[i] + ".png")
        worksheet.insert_image(chr(65 + column_number) + str("13"), "plot_non_dominated_" + algos[i] + ".png")
        column_number += 1

algos = ["M1_25_100_nsga2_heart_classic", "M1_50_100_nsga2_heart_classic", "M1_75_100_nsga2_heart_classic",
    "M1_25_1000_nsga2_heart_classic", "M1_50_1000_nsga2_heart_classic", "M1_75_1000_nsga2_heart_classic",
    "M1_25_10000_nsga2_heart_classic", "M1_50_10000_nsga2_heart_classic", "M1_75_10000_nsga2_heart_classic"]

t = 25

data = {'': ["precisão média", "tempo médio", "hipervolume médio", "AUC média", "desvio padrão médio da precisão", "desvio padrão médio do tempo", "", "melhor precisão", "melhor hipervolume", "melhor AUC"]}

problem = get_problem("SVM_hyperparameters_statlog", "resources/args_samples/M1_25_100.json")
problem.y_train = problem.y_train - 1

for algo in algos:
    print("on algo:", algo)

    avg_precision = 0
    avg_time = 0
    avg_hypervolume = 0
    avg_AUC = 0
    precisions = []
    times = []
    hypervolumes = []
    AUCs = []
    all_F = []

    for i in range(t):
        print("in:", i)

        this_F = []
        configs = []

        with open("results/" + algo + "/F" + str(i), 'r') as F_file:
            for line in F_file.readlines():
                F = json.loads(line)
                for f in F:
                    this_F.append(f)

        with open("results/" + algo + "/" + "P" + str(i), 'r') as decisionVariables_file:
            for line in decisionVariables_file.readlines():
                configs = json.loads(line)
                _, unique_indices = np.unique(this_F, return_index=True, axis=0)
                unique_configs = []
                for j in unique_indices:
                    unique_configs.append(configs[j])
                    all_F.append(this_F[j])
                configs = unique_configs

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

        # accuracy
        for config in configs:
            accuracy = problem.get_config_accuracy(config)
            precisions.append(accuracy)

        # AUC
        for config in configs:
            model = problem.get_config_model(config)
            model.fit(problem.X_train, problem.y_train)
            y_pred = model.decision_function(problem.X_train)

            fpr, tpr, thresholds = roc_curve(problem.y_train, y_pred)
            auc_value = auc(fpr, tpr)
            avg_AUC += auc_value
            AUCs.append(auc_value)

        print(configs)

    print("all F:", all_F)

    avg_precision /= len(precisions)
    avg_time /= t
    avg_hypervolume /= t
    avg_AUC /= t

    p_dp = 0
    t_dp = 0
    for i in range(len(precisions)):
        p_dp += (precisions[i] - avg_precision) ** 2
    for i in range(t):
        t_dp += (times[i] - avg_time) ** 2
    p_dp /= t
    t_dp /= t
    p_dp = math.sqrt(p_dp)
    t_dp = math.sqrt(t_dp)

    all_F = np.array(all_F)
    unique_all_F = np.unique(all_F, axis=0)
    plt.plot(unique_all_F[:,0], unique_all_F[:,1], 'bo')
    plt.savefig("plot_all_" + algo + ".png")
    plt.clf()

    bestPrecision, bestPrecision_i = get_best_value_and_index(precisions, maximize=True)
    bestHypervolume, _ = get_best_value_and_index(hypervolumes, maximize=True)
    bestAUC, bestAUC_i = get_best_value_and_index(AUCs, maximize=True)

    print("plotting")

    plot_colored_all(all_F, algo, bestPrecision_i, bestAUC_i)
    plot_colored_non_dominated(all_F, algo, bestPrecision_i, bestAUC_i)

    data[algo] = [avg_precision, avg_time, avg_hypervolume, avg_AUC, p_dp, t_dp, '', bestPrecision, bestHypervolume, bestAUC]

    print("")

df = pd.DataFrame(data)
writer = pd.ExcelWriter("demo.xlsx", engine="xlsxwriter")
df.to_excel(writer, sheet_name="Sheet1", index=False)

insert_plot_images(writer)

for column in df:
    column_length = max(df[column].astype(str).map(len).max(), len(column))
    col_idx = df.columns.get_loc(column)
    writer.sheets['Sheet1'].set_column(col_idx, col_idx, 45)

writer.close()