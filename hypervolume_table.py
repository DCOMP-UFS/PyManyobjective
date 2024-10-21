import json
import numpy as np
from deap.tools._hypervolume.pyhv import hypervolume
import math

def fill_algos_data(algos, problem, t, data):
    for algo in algos:
        print("on algo:", algo)

        avg_hypervolume = 0
        hypervolumes = []

        for i in range(t):
            with open("results/" + algo + "/F" + str(i), 'r') as F_file:
                for line in F_file.readlines():
                    F = json.loads(line)
                    F = np.array(F)
                    avg_hypervolume += hypervolume(np.copy(F), np.array([1.0, 1.0]))
                    hypervolumes.append(hypervolume(np.copy(F), np.array([1.0, 1.0])))

        avg_hypervolume /= t

        p_hv = 0
        for i in range(len(hypervolumes)):
            p_hv += (hypervolumes[i] - avg_hypervolume) ** 2
        p_hv /= len(hypervolumes)
        p_hv = math.sqrt(p_hv)

        data[algo] = {}
        data[algo]["hypervolume"] = avg_hypervolume
        data[algo]["std_hypervolume"] = p_hv

def generate_hypervolume_table(file_name, algos, problem, t, table_name="Null", label="null"):
    data = {}
    fill_algos_data(algos, problem, t, data)

    with open(file_name, "w", encoding='utf8') as table_file:
        table_file.write("\\begin{table}[]\n")
        table_file.write("\t\\centering\n")
        table_file.write("\t\\caption{" + table_name + "}\n")
        table_file.write("\t\\label{" + label + "}\n")
        table_file.write("\t\\begin{tabular}{c|c|c|c}\n")
        table_file.write("\t\\hline\n")
        table_file.write("\tAlgoritmo & Num. avaliações & Média Hipervolume & Std. Hipervolume \\\\\n")
        table_file.write("\t\\hline\n")

        for algo in algos:
            algo_name = ""
            n_evaluations = "0"
            algo_splitted = algo.split("_")
            if algo_splitted[0] == "M1":
                algo_name = "M1 " + algo_splitted[1] + "\\%"
                n_evaluations = algo_splitted[2]
            elif algo_splitted[0] == "DVL":
                algo_name = "DVL"
                n_evaluations = algo_splitted[1] 
            elif algo_splitted[0] == "TDEAP":
                algo_name = "TDEAP"
                n_evaluations = algo_splitted[1] 
            else:
                algo_name = "NSGA-II"
                n_evaluations = algo_splitted[1]

            line_content = algo_name + " & " + n_evaluations 
            line_content += " & " + str(round(data[algo]["hypervolume"], 4))
            line_content += " & " + str(round(data[algo]["std_hypervolume"], 4))

            table_file.write("\t")
            table_file.write(line_content)
            table_file.write(" \\\\\n")

        table_file.write("\t\\end{tabular}\n")
        table_file.write("\\end{table}")
