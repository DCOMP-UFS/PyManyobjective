from boxplot import generate_boxplot

algos = ["DVL_100", "M1_25_100_nsga2", "NSGA-II_100", 'TDEAP_100',
        "DVL_500", "M1_25_500_nsga2", "NSGA-II_500", 'TDEAP_500',]

t = 25

generate_boxplot(algos, t)