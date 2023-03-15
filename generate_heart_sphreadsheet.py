from sphreadsheet import *
from run_hyper import get_problem

algos = ["M1_25_100_nsga2_heart", "M1_50_100_nsga2_heart", "M1_75_100_nsga2_heart", "NSGAII_100_heart",
    "M1_25_1000_nsga2_heart", "M1_50_1000_nsga2_heart", "M1_75_1000_nsga2_heart", "NSGAII_1000_heart",
    "M1_25_10000_nsga2_heart", "M1_50_10000_nsga2_heart", "M1_75_10000_nsga2_heart", "NSGAII_10000_heart"]

t = 25

problem = get_problem("SVM_hyperparameters_statlog", "resources/args_samples/M1_25_100.json")
problem.y_train = problem.y_train - 1

generate_sphreadsheet("heart", algos, problem, t)