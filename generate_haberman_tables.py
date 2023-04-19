import math
import pandas as pd
import json
import numpy as np
from run_hyper import get_problem

from class_table import generate_class_table
from hypervolume_table import generate_hypervolume_table
from time_table import generate_time_table

algos = ["M1_25_100_nsga2_haberman", "M1_50_100_nsga2_haberman", "M1_75_100_nsga2_haberman", "NSGAII_100_haberman",
    "M1_25_1000_nsga2_haberman", "M1_50_1000_nsga2_haberman", "M1_75_1000_nsga2_haberman", "NSGAII_1000_haberman",
    "M1_25_10000_nsga2_haberman", "M1_50_10000_nsga2_haberman", "M1_75_10000_nsga2_haberman", "NSGAII_10000_haberman"]

t = 25

problem = get_problem("SVM_hyperparameters_haberman", "resources/args_samples/M1_25_100.json")

generate_hypervolume_table("haberman_hypervolume_table.txt", algos, problem, t, table_name="Hipervolume - Haberman's Survival", label="tab:hvhaberman")
generate_time_table("haberman_time_table.txt", algos, problem, t, table_name="Tempo - Haberman's Survival", label="tab:timehaberman")
generate_class_table("haberman_class_table.txt", algos, problem, t, table_name="Class - Haberman's Survival", label="tab:classhaberman")