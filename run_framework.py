from run import run

run("M1", "NN_MNIST", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1, 
    framework_args_file="resources/args_samples/framework_args_M1.json", 
    moea_args_file="resources/args_samples/MEMO_args.json", 
    pareto_front_file="tests/dtlz1_front.txt", 
    problem_args_file="resources/args_samples/problem_args_DTLZ.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json")
