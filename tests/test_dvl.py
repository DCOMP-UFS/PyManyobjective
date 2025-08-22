from src.MOEAs import NSGAII
from run import get_mutation, get_crossover, get_selection, get_sparsity
from src.problems.DTLZ import DTLZ2
from src.dvl.DLV import DVLFramework
from src.MOEAs.NSGAIII import NSGAIII
from src.dvl.models.Linear import LinearModel as LR

from scipy.stats.qmc import LatinHypercube, QMCEngine

from src.problems.Problem import Problem
from src.MOEAs.Algorithm import Algorithm
from scipy.stats.qmc import LatinHypercube, QMCEngine
from src.Solution import Solution


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from src.problems import DTLZ
from src.MOEAs import NSGAII
from run import get_mutation, get_crossover, get_selection, get_sparsity
from src.ParetoFront import ParetoFront
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def plota_3d(pontos, n_obj, evals_allowed, pop_size):
    x_coords = [p[0] for p in pontos]
    y_coords = [p[1] for p in pontos]
    z_coords = [p[2] for p in pontos]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=45) # type: ignore
    ax.scatter(x_coords, y_coords, z_coords, c='#0073CF')
    title = f"DVL-Framework - num_objetivos = {n_obj}, avaliações = {evals_allowed} e tamanho_população = {pop_size}"
    ax.set_title(title)
    plt.show()


def run(n_obj, evals_allowed, pop_size):
    """ pg. 82
        3 objectives and
        12 variables and the other with 10 objectives and 12 variables

        pg.85 section 4.2.2
        Each algorithm runs until reaches a maximum number of evaluations, defined as
        (250, 500, 1000, 1500, 10000, 100000)

        Hypervolume (HV) (WHILE et al., 2006), a performance indicator
        that assesses both convergence and divergence of the solutions

        pg. 85 table 18
        DVL parameter configuration for the optimization experiment
    """
# >> Config Vars
    """ pg.86
        For all the algorithms the population size used
        was 91 and 220 for the problem with 3 and 10 objectives, respectively, and the individual has a
        length of 12 for both objective sizes
    """
    # n_obj = 3 #usaremos 3 e 10 
    # evals_allowed = 10000 # usaremos 1000, 10000 e 100000
    # pop_size = 100 # usaremos 100, 150 e 200
    
    
    
    n_var = 12 # provavelmente devo manter esses 12
    ProblemClass = DTLZ2
# << Config Vars


    # does this class hold state? just in case will create 2
    dvl_problem   =  ProblemClass(n_obj,k=n_var-n_obj+1)

    """ pg. 86
        For the NSGA-III, using simulated binary
        crossover (DEB; AGRAWAL et al., 1995), the distribution index is set to [2 = 30, and the
        crossover probability ?2 = 1.0. 
    """

    dvl_moea  = NSGAIII
    pure_moea = NSGAIII

    """@obs
        Linear Regression and other scikit learn need to be pipelined?
        >> might need a bit for experience with scikit learn models
    """
    model = LR()
    framework = DVLFramework( 
        pop_size=pop_size, 
        max_eval=evals_allowed, 
        ClassMoea=dvl_moea, 
        model=model, 
        problem=dvl_problem
    )
    

    # framework.moea = nsga
    P = framework.execute()
    P_objective = [list(map(float, elem.objectives)) for elem in P]
    print(P_objective)
    plota_3d(P_objective, n_obj, evals_allowed, pop_size)
    # Retirar os objetivos dessa população P para jogar pro plot 3D e imprimir os objetivos.
    # A dificuldade está sendo acessar os objetivos de P. o P.objectives não esta funcionando,
    # Ta dando isso: AttributeError: 'set' object has no attribute 'objectives'
    return P_objective
    

    """pg. 87
        DVL Framework outperformed the MOEAs especially in the initial
        evaluations, approaching zero in the difference of hypervolume when it reaches 100000 objective
        function evaluations. 
    """


    # nsga = NSGAII.NSGAII(dvl_problem, evals_allowed, pop_size, pop_size, None, None, None, None)
    # nsga.mutation = get_mutation("Polynomial", "resources/args_samples/Polynomial_args.json")
    # nsga.crossover = get_crossover("SBX", "resources/args_samples/SBX_args.json")
    # nsga.selection = get_selection("Binary")
    # nsga.sparsity = get_sparsity("CrowdingDistance")
    
# DVL-Framework - num_objetivos = 3, avaliações = 10000 e tamanho_população = 200 

# Para 3 objetivos: 1000, 10000 e 100000
    # pop_size = 100, 150, 200 

# Para 3 objetivos: 1000, 10000 e 100000
    # pop_size = 100, 150, 200

# 10 execuções:
    # pop_size=100, max_eval=1000, obj=3
    # pop_size=150, max_eval=1000, obj=3
    # pop_size=200, max_eval=1000, obj=3

    # pop_size=100, max_eval=10000, obj=3
    # pop_size=150, max_eval=10000, obj=3
    # pop_size=200, max_eval=10000, obj=3
    
    # pop_size=100, max_eval=100000, obj=3
    # pop_size=150, max_eval=100000, obj=3
    # pop_size=200, max_eval=100000, obj=3


    # pop_size=100, max_eval=1000, obj=10
    # pop_size=150, max_eval=1000, obj=10
    # pop_size=200, max_eval=1000, obj=10

    # pop_size=100, max_eval=10000, obj=10
    # pop_size=150, max_eval=10000, obj=10
    # pop_size=200, max_eval=10000, obj=10
    
    # pop_size=100, max_eval=100000, obj=10
    # pop_size=150, max_eval=100000, obj=10
    # pop_size=200, max_eval=100000, obj=10










# Pegar os 10 igd's dessas execução
# DTLZ1_M1_100
# DTLZ1_M3_100
# DTLZ1_M6_100

# DTLZ2_M1_100
# DTLZ2_M3_100
# DTLZ2_M6_100


# Pegar os 10 igd's dessas execuções
# DTLZ1_M1_1000 
# DTLZ1_M3_1000
# DTLZ1_M6_1000

# DTLZ2_M1_1000
# DTLZ2_M3_1000
# DTLZ2_M6_1000


# Pegar os 10 igd's dessas execução
# DTLZ1_M1_10000
# DTLZ1_M3_10000
# DTLZ1_M6_10000

# DTLZ2_M1_10000
# DTLZ2_M3_10000
# DTLZ2_M6_10000