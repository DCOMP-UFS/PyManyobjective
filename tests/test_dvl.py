from src.problems.DTLZ import DTLZ2
from src.dvl.DLV import DVLFramework, save_obj_space
from src.MOEAs.NSGAII import NSGAII
from src.dvl.models.Linear import LinearModel as LR

from scipy.stats.qmc import LatinHypercube, QMCEngine

from src.problems.Problem import Problem
from src.MOEAs.Algorithm import Algorithm
from scipy.stats.qmc import LatinHypercube, QMCEngine
from src.Solution import Solution

def run():
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
    n_var = 12
    n_obj = 10

    evals_allowed = 250
    pop_size    = 100
    ProblemClass = DTLZ2
# << Config Vars


    # does this class hold state? just in case will create 2
    dvl_problem   =  ProblemClass(n_obj,k=n_var-n_obj+1)

    """ pg. 86
        For the NSGA-III, using simulated binary
        crossover (DEB; AGRAWAL et al., 1995), the distribution index is set to [2 = 30, and the
        crossover probability ?2 = 1.0. 
    """

    dvl_moea  = NSGAII
    pure_moea = NSGAII

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

    framework.execute()
    
    #save_obj_space("teste_fig", framework.moea.population.objectives, None)
    """pg. 87
        DVL Framework outperformed the MOEAs especially in the initial
        evaluations, approaching zero in the difference of hypervolume when it reaches 100000 objective
        function evaluations. 
    """
