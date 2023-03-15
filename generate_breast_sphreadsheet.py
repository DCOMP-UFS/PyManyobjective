from sphreadsheet import *
from run_hyper import get_problem

algos = ["M1_25_100_nsga2_breast", "M1_50_100_nsga2_breast", "M1_75_100_nsga2_breast", "NSGAII_100_breast",
    "M1_25_1000_nsga2_breast", "M1_50_1000_nsga2_breast", "M1_75_1000_nsga2_breast", "NSGAII_1000_breast",
    "M1_25_10000_nsga2_breast", "M1_50_10000_nsga2_breast", "M1_75_10000_nsga2_breast", "NSGAII_10000_breast"]

t = 25

problem = get_problem("SVM_hyperparameters_breast", "resources/args_samples/M1_25_100.json")

generate_sphreadsheet("breast", algos, problem, t)