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
plt.figure(figsize=[3.5, 3.0])

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

    color_values = ["black", "green", "orange", "blue"]
    colors = np.zeros((F.shape[0]), dtype=np.int32)
    colors[bestPrecision_i] = 1
    colors[bestAUC_i] = 2
    if bestPrecision_i == bestAUC_i:
        colors[bestPrecision_i] = 3
    colors = colors.tolist()

    F = -F + 1.0

    colored_x = []
    colored_y = []
    colored_color = []

    for i in range(len(F)):
        if color_values[colors[i]] == 0:
            plt.scatter(F[:,0][i], F[:,1][i], c=color_values[colors[i]])
        else:
            colored_x.append(F[:,0][i])
            colored_y.append(F[:,1][i])
            colored_color.append(colors[i])
    for i in range(len(colored_x)):
        plt.scatter(colored_x[i], colored_y[i], c=color_values[colored_color[i]])

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

    color_values = ["black", "green", "orange", "red"]
    colors = np.zeros((F.shape[0]), dtype=np.int32)
    colors[bestPrecision_i] = 1
    colors[bestAUC_i] = 2
    if bestPrecision_i == bestAUC_i:
        colors[bestPrecision_i] = 3
    colors[fronts == 0]
    colors = colors.tolist()

    non_dominated_F = F[fronts == 0]
    non_dominated_F = -non_dominated_F + 1.0

    colored_x = []
    colored_y = []
    colored_color = []

    for i in range(len(non_dominated_F)):
        if color_values[colors[i]] == 0:
            plt.scatter(non_dominated_F[:,0][i], non_dominated_F[:,1][i], c=color_values[colors[i]])
        else:
            colored_x.append(non_dominated_F[:,0][i])
            colored_y.append(non_dominated_F[:,1][i])
            colored_color.append(colors[i])
    for i in range(len(colored_x)):
        plt.scatter(colored_x[i], colored_y[i], c=color_values[colored_color[i]])

    plt.savefig("plot_non_dominated_" + algo_name + ".png")
    plt.clf()

def plot_ROC_of_best_AUC(config, problem, algo_name):
    model = problem.get_config_model(config)
    model.fit(problem.X_train, problem.y_train)
    y_pred = model.decision_function(problem.X_train)

    fpr, tpr, thresholds = roc_curve(problem.y_train, y_pred)
    plt.grid()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    
    plt.savefig("plot_roc_" + algo_name + ".png")
    plt.clf()

def insert_plot_images(writer, algos):
    worksheet = writer.sheets["Sheet1"]
    worksheet.set_row(12, 220)
    worksheet.set_row(13, 220)
    worksheet.set_row(14, 220)
    column_number = 1
    for i in range(len(algos)):
        worksheet.insert_image(chr(65 + column_number) + str("13"), "plot_all_" + algos[i] + ".png")
        worksheet.insert_image(chr(65 + column_number) + str("14"), "plot_non_dominated_" + algos[i] + ".png")
        worksheet.insert_image(chr(65 + column_number) + str("15"), "plot_roc_" + algos[i] + ".png")
        column_number += 1

def GRA(f):
    f = -f + 1.0
    F = (f - np.min(f, axis=0)) / np.ptp(f, axis=0)
    F_p = np.max(F, axis=0)
    delta_I = np.abs(F - F_p)
    delta_min = np.min(delta_I)
    delta_max = np.max(delta_I)
    GRC = (1 / F.shape[0]) * np.sum((delta_min + delta_max) / (delta_I + delta_max), axis=1)
    return np.argmax(GRC)

def fill_algos_data(algos, problem, t, data):
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
        all_configs = []

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
                avg_precision += accuracy

            # AUC
            for config in configs:
                model = problem.get_config_model(config)
                model.fit(problem.X_train, problem.y_train)
                y_pred = model.decision_function(problem.X_train)

                fpr, tpr, thresholds = roc_curve(problem.y_train, y_pred)
                auc_value = auc(fpr, tpr)
                avg_AUC += auc_value
                AUCs.append(auc_value)

                all_configs.append(config)

            print(configs)

        print("all F:", all_F)

        avg_precision /= len(precisions)
        avg_time /= t
        avg_hypervolume /= t
        avg_AUC /= len(AUCs)

        p_dp = 0
        t_dp = 0
        for i in range(len(precisions)):
            p_dp += (precisions[i] - avg_precision) ** 2
        for i in range(t):
            t_dp += (times[i] - avg_time) ** 2
        p_dp /= len(precisions)
        t_dp /= t
        p_dp = math.sqrt(p_dp)
        t_dp = math.sqrt(t_dp)

        p_hv = 0
        for i in range(len(hypervolumes)):
            p_hv += (hypervolumes[i] - avg_hypervolume) ** 2
        p_hv /= len(hypervolumes)
        p_hv = math.sqrt(p_hv)

        bestPrecision, bestPrecision_i = get_best_value_and_index(precisions, maximize=True)
        bestHypervolume, _ = get_best_value_and_index(hypervolumes, maximize=True)
        bestAUC, bestAUC_i = get_best_value_and_index(AUCs, maximize=True)

        print("plotting")

        all_F = np.array(all_F)

        best_MO_i = GRA(np.copy(all_F))
        bestMOPrecision = precisions[best_MO_i]
        bestMOAUC = AUCs[best_MO_i]

        plot_ROC_of_best_AUC(all_configs[bestAUC_i], problem, algo)
        plot_colored_all(np.copy(all_F), algo, bestPrecision_i, bestAUC_i)
        plot_colored_non_dominated(np.copy(all_F), algo, bestPrecision_i, bestAUC_i)

        data[algo] = [avg_precision, avg_time, avg_hypervolume, avg_AUC, p_dp, t_dp, p_hv, bestPrecision, bestHypervolume, bestAUC, bestMOPrecision, bestMOAUC]

        print("")

def generate_sphreadsheet(file_name, algos, problem, t):
    data = {'': ["precisão média", "tempo médio", "hipervolume médio", "AUC média", "desvio padrão médio da precisão", "desvio padrão médio do tempo", "desvio padrão do hipervolume", "melhor precisão", "melhor hipervolume", "melhor AUC", "melhor precisão GRA", "melhor AUC GRA"]}
    fill_algos_data(algos, problem, t, data)

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(file_name + ".xlsx", engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1", index=False)

    insert_plot_images(writer, algos)

    for column in df:
        column_length = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, 45)

    writer.close()