from run_hyper import run

#run("M3", "DTLZ2", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
#    1, 
#    args_file="resources/args_samples/M3_25_100.json", )

run("M1", "SVM_hyperparameters_breast", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    25,
    args_file="resources/args_samples/M1_25_500.json",
    save_dir="M1_25_500_nsga2")



'''run("Dummy", "SVM_hyperparameters_breast", "NSGAII", "SBX", "Polynomial", "Binary", "CrowdingDistance",
    25,
    args_file="resources/args_samples/NSGA2_500.json",
    save_dir="NSGA2_500")'''