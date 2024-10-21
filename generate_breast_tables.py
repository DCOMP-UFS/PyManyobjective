import math
import pandas as pd
import json
import numpy as np
from run_hyper import get_problem

from class_table import generate_class_table
from hypervolume_table import generate_hypervolume_table
from time_table import generate_time_table
'''
algos = ["M1_25_100_nsga2_breast", "M1_50_100_nsga2_breast", "M1_75_100_nsga2_breast", "NSGAII_100_breast",
    "M1_25_1000_nsga2_breast", "M1_50_1000_nsga2_breast", "M1_75_1000_nsga2_breast", "NSGAII_1000_breast",
    "M1_25_10000_nsga2_breast", "M1_50_10000_nsga2_breast", "M1_75_10000_nsga2_breast", "NSGAII_10000_breast"]
'''

algos = ["DVL_100", "M1_25_100_nsga2", "NSGA-II_100", 'TDEAP_100',
        "DVL_500", "M1_25_500_nsga2", "NSGA-II_500", 'TDEAP_500',]

t = 25

problem = get_problem("SVM_hyperparameters_breast", "resources/args_samples/M1_25_1000.json")

generate_hypervolume_table("breast_hv_table.txt", algos, problem, t, table_name="Hipervolume - Breast Cancer", label="tab:hvbreast")
generate_time_table("breast_time_table.txt", algos, problem, t, table_name="Tempo - Breast Cancer", label="tab:timebreast")
generate_class_table("breast_class_table.txt", algos, problem, t, table_name="Class - Breast Cancer", label="tab:classbreast")