from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from src.problems import DTLZ
from src.MOEAs import NSGAII
from run import get_mutation, get_crossover, get_selection, get_sparsity
from src.ParetoFront import ParetoFront

def plota_3d(pontos, title=""):
    x_coords = [p[0] for p in pontos]
    y_coords = [p[1] for p in pontos]
    z_coords = [p[2] for p in pontos]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=45)
    ax.scatter(x_coords, y_coords, z_coords, c='#0073CF')
    ax.set_title(title)
    plt.show()

def plota_2d(pontos, title=""):
    x_coords = [p[0] for p in pontos]
    y_coords = [p[1] for p in pontos]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(x_coords, y_coords, c='#0073CF')
    ax.set_title(title)
    plt.show()

problem = DTLZ.DTLZ2(numberOfObjectives=3, k=12)
nsga = NSGAII.NSGAII(problem, 1_000, 200, 200, None, None, None, None)
nsga.mutation = get_mutation("Polynomial", "resources/args_samples/Polynomial_args.json")
nsga.crossover = get_crossover("SBX", "resources/args_samples/SBX_args.json")
nsga.selection = get_selection("Binary")
nsga.sparsity = get_sparsity("CrowdingDistance")

nsga.execute()
p = nsga.population
f = ParetoFront()
a = f.fastNonDominatedSort(p)
p.filter(a==0)
plota_3d(p.objectives.tolist(), "NSGA-II")
