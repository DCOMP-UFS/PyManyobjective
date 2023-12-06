from run import run

run("SMO", "DTLZ1", "MRGA", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    1, 
    framework_args_file="resources/args_samples/framework_args_SOM.json", 
    moea_args_file="resources/args_samples/MEMO_args.json", 
    pareto_front_file="tests/dtlz1_front.txt", 
    problem_args_file="resources/args_samples/problem_args_DTLZ.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json") 

