from wilcoxon_heatmap import generate_wilcoxon_heatmap

algos = ["M1_25_100_nsga2_breast", "M1_50_100_nsga2_breast", "M1_75_100_nsga2_breast", "NSGAII_100_breast",
    "M1_25_1000_nsga2_breast", "M1_50_1000_nsga2_breast", "M1_75_1000_nsga2_breast", "NSGAII_1000_breast",
    "M1_25_10000_nsga2_breast", "M1_50_10000_nsga2_breast", "M1_75_10000_nsga2_breast", "NSGAII_10000_breast"]

t = 25

generate_wilcoxon_heatmap(algos, t)