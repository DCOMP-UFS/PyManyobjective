from run_hyper import run

run("M1_linear", "SVM_hyperparameters_statlog", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    25,
    framework_args_file="resources/args_samples/framework_args_M1.json",
    moea_args_file="resources/args_samples/MEMO_args.json",
    pareto_front_file="tests/zdt1_front.txt",
    problem_args_file="resources/args_samples/problem_args_ZDT.json",
    mutation_args_file="resources/args_samples/Polynomial_args.json",
    crossover_args_file="resources/args_samples/SBX_args.json",
    selection_args_file="resources/args_samples/Bin_args.json",
    save_dir="M1_25_1000_nsga2")
