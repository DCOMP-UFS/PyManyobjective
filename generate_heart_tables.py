import math
import pandas as pd
import json
import numpy as np
from run_hyper import get_problem

from class_table import generate_class_table
from hypervolume_table import generate_hypervolume_table
from time_table import generate_time_table

algos = ["M1_25_100_nsga2_heart", "M1_50_100_nsga2_heart", "M1_75_100_nsga2_heart", "NSGAII_100_heart",
    "M1_25_1000_nsga2_heart", "M1_50_1000_nsga2_heart", "M1_75_1000_nsga2_heart", "NSGAII_1000_heart",
    "M1_25_10000_nsga2_heart", "M1_50_10000_nsga2_heart", "M1_75_10000_nsga2_heart", "NSGAII_10000_heart"]

t = 25

problem = get_problem("SVM_hyperparameters_statlog", "resources/args_samples/M1_25_100.json")
problem.y_train = problem.y_train - 1

generate_hypervolume_table("heart_hypervolume_table.txt", algos, problem, t, table_name="Hipervolume - Heart Disease", label="tab:hvheart")
generate_time_table("heart_time_table.txt", algos, problem, t, table_name="Tempo - Heart Disease", label="tab:timeheart")
generate_class_table("heart_class_table.txt", algos, problem, t, table_name="Class - Heart Disease", label="tab:classheart")