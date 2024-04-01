#from run import run
from graph_plot import plot_from_file

""" run("Dummy", "DTLZ2", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1,
    args_file="resources/args_samples/NSGA2_1000.json") 

plot_from_file('Dummy_DTLZ2_NSGAII') 

run("M6", "DTLZ2", "MRGA", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1, 
    args_file="resources/args_samples/M6_1000.json",
    ) """

from run2 import run 

run("M1", "NN_MNIST", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1, 
    framework_args_file="resources/args_samples/framework_args_M1.json", 
    moea_args_file="resources/args_samples/MEMO_args.json", 
    pareto_front_file="tests/dtlz1_front.txt", 
    problem_args_file="resources/args_samples/problem_args_DTLZ.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json")