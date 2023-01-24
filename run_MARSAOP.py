import warnings
from run_hyper import run

warnings.simplefilter(action='ignore', category=FutureWarning)

run("new_MARSAOP", "SVM_hyperparameters_boston", "GA", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    25,
    framework_args_file="resources/args_samples/framework_args_new_MARSAOP.json",
    moea_args_file="resources/args_samples/MEMO_args.json",
    pareto_front_file="tests/zdt1_front.txt",
    problem_args_file="resources/args_samples/problem_args_ZDT.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json")
