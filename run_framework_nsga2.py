from run_hyper import run

run("M1_linear", "SVM_hyperparameters_statlog", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    25,
    args_file="resources/args_samples/M1_75_1000.json",
    save_dir="M1_75_1000_nsga2_breast_classic")
