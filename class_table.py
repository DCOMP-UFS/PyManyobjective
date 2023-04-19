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
        avg_AUC = 0
        avg_F_measure = 0
        precisions = []
        AUCs = []
        F_measures = []
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

            best_MO_i = GRA(np.array(this_F))
            print(best_MO_i)

            with open("results/" + algo + "/" + "P" + str(i), 'r') as decisionVariables_file:
                for line in decisionVariables_file.readlines():
                    configs = json.loads(line)
                    for i in range(len(configs)):
                        if i == best_MO_i:
                            configs = [configs[i]]
                            break

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

            # F-measure
            for config in configs:
                F_measure = problem.get_config_f_measure(config)
                F_measures.append(F_measure)
                avg_F_measure += F_measure

        avg_precision /= len(precisions)
        avg_AUC /= len(AUCs)
        avg_F_measure /= len(F_measures)

        p_dp = 0
        for i in range(len(precisions)):
            p_dp += (precisions[i] - avg_precision) ** 2
        p_dp /= len(precisions)
        p_dp = math.sqrt(p_dp)

        #bestPrecision, bestPrecision_i = get_best_value_and_index(precisions, maximize=True)
        #bestAUC, bestAUC_i = get_best_value_and_index(AUCs, maximize=True)

        #all_F = np.array(all_F)

        #best_MO_i = GRA(np.copy(all_F))
        #bestMOPrecision = precisions[best_MO_i]
        #bestMOAUC = AUCs[best_MO_i]

        data[algo] = {}
        data[algo]["avg_accuracy"] = avg_precision
        data[algo]["avg_AUC"] = avg_AUC
        data[algo]["std_accuracy"] = p_dp
        data[algo]["avg_F_measure"] = avg_F_measure

def generate_class_table(file_name, algos, problem, t, table_name="Null", label="null"):
    data = {}
    fill_algos_data(algos, problem, t, data)

    with open(file_name, "w") as table_file:
        table_file.write("\\begin{table}[]\n")
        table_file.write("\t\\centering\n")
        table_file.write("\t\\caption{" + table_name + "}\n")
        table_file.write("\t\\label{" + label + "}\n")
        table_file.write("\t\\begin{tabular}{c|c|c|c|c}\n")
        table_file.write("\t\\hline\n")
        table_file.write("\tAlgoritmo & Num. avaliações & Média Precisão & Std. Precisão & Média AUC & Média F-measure \\\\\n")
        table_file.write("\t\\hline\n")

        for algo in algos:
            algo_name = ""
            n_evaluations = "0"
            algo_splitted = algo.split("_")
            if algo_splitted[0] == "M1":
                algo_name = "M1 " + algo_splitted[1] + "\\%"
                n_evaluations = algo_splitted[2]
            else:
                algo_name = "NSGA-II"
                n_evaluations = algo_splitted[1]

            line_content = algo_name + " & " + n_evaluations 
            line_content += " & " + str(round(data[algo]["avg_accuracy"], 4))
            line_content += " & " + str(round(data[algo]["std_accuracy"], 4))
            line_content += " & " + str(round(data[algo]["avg_AUC"], 4))
            line_content += " & " + str(round(data[algo]["avg_F_measure"], 4))

            table_file.write("\t")
            table_file.write(line_content)
            table_file.write(" \\\\\n")

        table_file.write("\t\\end{tabular}\n")
        table_file.write("\\end{table}")
