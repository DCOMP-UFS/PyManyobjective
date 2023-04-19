from wilcoxon_heatmap import generate_wilcoxon_heatmap

algos = ["M1_25_100_nsga2_heart", "M1_50_100_nsga2_heart", "M1_75_100_nsga2_heart", "NSGAII_100_heart",
    "M1_25_1000_nsga2_heart", "M1_50_1000_nsga2_heart", "M1_75_1000_nsga2_heart", "NSGAII_1000_heart",
    "M1_25_10000_nsga2_heart", "M1_50_10000_nsga2_heart", "M1_75_10000_nsga2_heart", "NSGAII_10000_heart"]

t = 25

generate_wilcoxon_heatmap(algos, t)