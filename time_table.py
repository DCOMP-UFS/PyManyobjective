import json
import numpy as np
from deap.tools._hypervolume.pyhv import hypervolume
import math

def fill_algos_data(algos, problem, t, data):
    for algo in algos:
        print("on algo:", algo)

        avg_time = 0
        times = []

        for i in range(t):
            with open("results/" + algo + "/T" + str(i), 'r') as time_file:
                for line in time_file.readlines():
                    avg_time += float(line)
                    times.append(float(line))

        avg_time /= t

        p_time = 0
        for i in range(len(times)):
            p_time += (times[i] - avg_time) ** 2
        p_time /= len(times)
        p_time = math.sqrt(p_time)

        data[algo] = {}
        data[algo]["time"] = avg_time
        data[algo]["std_time"] = p_time

def generate_time_table(file_name, algos, problem, t, table_name="Null", label="null"):
    data = {}
    fill_algos_data(algos, problem, t, data)

    with open(file_name, "w", encoding='utf8') as table_file:
        table_file.write("\\begin{table}[]\n")
        table_file.write("\t\\centering\n")
        table_file.write("\t\\caption{" + table_name + "}\n")
        table_file.write("\t\\label{" + label + "}\n")
        table_file.write("\t\\begin{tabular}{c|c|c|c}\n")
        table_file.write("\t\\hline\n")
        table_file.write("\tAlgoritmo & Num. avaliações & Média Tempo & Std. Tempo \\\\\n")
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
            line_content += " & " + str(round(data[algo]["time"], 4))
            line_content += " & " + str(round(data[algo]["std_time"], 4))

            table_file.write("\t")
            table_file.write(line_content)
            table_file.write(" \\\\\n")

        table_file.write("\t\\end{tabular}\n")
        table_file.write("\\end{table}")
