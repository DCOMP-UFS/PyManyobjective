from sphreadsheet import *
from run_hyper import get_problem

algos = ["M1_25_100_nsga2_haberman", "M1_50_100_nsga2_haberman", "M1_75_100_nsga2_haberman", "NSGAII_100_haberman",
    "M1_25_1000_nsga2_haberman", "M1_50_1000_nsga2_haberman", "M1_75_1000_nsga2_haberman", "NSGAII_1000_haberman",
    "M1_25_10000_nsga2_haberman", "M1_50_10000_nsga2_haberman", "M1_75_10000_nsga2_haberman"]

t = 25

problem = get_problem("SVM_hyperparameters_haberman", "resources/args_samples/M1_25_100.json")

generate_sphreadsheet("haberman", algos, problem, t)