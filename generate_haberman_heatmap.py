from wilcoxon_heatmap import generate_wilcoxon_heatmap

algos = ["M1_25_100_nsga2_haberman", "M1_50_100_nsga2_haberman", "M1_75_100_nsga2_haberman", "NSGAII_100_haberman",
    "M1_25_1000_nsga2_haberman", "M1_50_1000_nsga2_haberman", "M1_75_1000_nsga2_haberman", "NSGAII_1000_haberman",
    "M1_25_10000_nsga2_haberman", "M1_50_10000_nsga2_haberman", "M1_75_10000_nsga2_haberman", "NSGAII_10000_haberman"]

t = 25

generate_wilcoxon_heatmap(algos, t)