from run import run

run("Dummy", "DTLZ2", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1,
    args_file="resources/args_samples/NSGA2_1000.json") 

generate_boxplot(['Dummy_DTLZ2_NSGAII'], 1)

""" run("Dummy", "DTLZ2", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1, 
    framework_args_file=None, 
    moea_args_file="resources/args_samples/MEMO_args.json", 
    pareto_front_file="tests/dtlz1_front.txt", 
    problem_args_file="resources/args_samples/problem_args_DTLZ.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json") """ 

""" from src.problems import DTLZ
from src.MOEAs import NSGAII
from src.frameworks.Dummy import Dummy
from run import get_mutation, get_crossover, get_selection, get_sparsity

problem = DTLZ.DTLZ2(numberOfObjectives=3, k=12)
nsga = NSGAII.NSGAII(problem, 1_000, 200, 200, None, None, None, None)
nsga.mutation = get_mutation("Polynomial", "resources/args_samples/Polynomial_args.json")
nsga.crossover = get_crossover("SBX", "resources/args_samples/SBX_args.json")
nsga.selection = get_selection("Binary")
nsga.sparsity = get_sparsity("CrowdingDistance")

d = Dummy(problem, nsga)

print(d.run().objectives.tolist())
print("\n\nTESTE\n\n",nsga.population.objectives.tolist())
 """