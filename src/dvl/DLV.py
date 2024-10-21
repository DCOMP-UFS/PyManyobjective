import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache

from copy import deepcopy
from scipy.stats.qmc import LatinHypercube, QMCEngine
from src.Solution import Solution
from src.Population import Population, genPopulation
from src.BinaryTournament import BinaryTournament
from src.MOEAs.crossovers.SBXCrossover import SBXCrossover
from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation
from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance
from src.problems.Problem import Problem
from src.MOEAs.Algorithm import Algorithm
from src.dvl.Model import Model
from src.ParetoFront import ParetoFront


class DVLFramework:
    # for a problem with 'n' variables, '1' supervised learning models will be used
    def __init__(self,
            pop_size:int, # k, initial popsize
            max_eval:int, # e, max number of objective evaluation 
            ClassMoea, 
            model: Model, 
            problem: Problem
        ):
        self.pop_size = pop_size
        self.max_eval = max_eval
        self.problem: Problem = problem
        self.model: Model = model
        self.moea = None
        """@obs
            API needed to be adapted because we can't determine the max eval upfront.
            we need to pass the moea class and construct it on the fly, to delay the 
            knowledge of max_eval value; 
        """
        self.ClassMoea = ClassMoea



    def execute(self):
        sampling: QMCEngine = LatinHypercube(
            d=self.problem.numberOfDecisionVariables
        )
        k:int = self.pop_size
        e:int = self.max_eval
        P_est, objectives = self.execute_dvl(sampling=sampling, dataset_size=k)
        #print(f'\n\n\n\n{P_est} {objectives}\n\n\n')
        # 'd' evaluations left (calls to objective functions) to complete the framework
        d = e - (k + len(P_est.decisionVariables))
        P = self.execute_moea(P_est, objectives, e, d)
        return P


    # DVL Inverse Modeling only
    # pg. 80 "The necessity of using the hypervolume inside the algorithm is eliminated"
    def execute_dvl(self, sampling: QMCEngine, dataset_size: int, reference_points=None):
        Pt = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        Pk = genPopulation(self.problem, samples=100)
        
        #print("objs", Pk.objectives)
        """@obs
            .evaluate doesnt take a array as input
            also it doesnt return objetives, it returns the same solution 
            from which i should extract the objetives on the from the variable
            .objectives 

            class Solution() needed to be changed because we can't pass a already
            valid solution as np.array to the constructor 
        """

        if reference_points is None:
            """"@obs
                Genrate_reference takes too long to iterate over, 
                in the paper it seems that it uses a set a references points already baked
            """
            x = 0
            y = 1
            arr = []
            arr.append((1,0))
            for _ in range(50):
                arr.append((x,y))
                x += 0.02
                y = (1-(x*x)) ** 0.5
                

            reference_points = np.asarray( arr
                #generate_reference_points(n_obj=self.problem.numberOfObjectives, n_div_per_obj=2)
            )
        
        """@obs 
            It seems that in the code we use only 1 (one) model
            in case it's a list we use
        for M in self.models:
            # Each model is gi(Y) = xi or we can make one model g(Y) = X, correct?
            M.train(solutions, objectives)
        """
        # ONE model is g(Y) = X, correct?
        #print("Ref", reference_points)
        self.model.train(Pk.decisionVariables, Pk.objectives)

        # Estimation for a population as close as possible to the Pareto-optimal front
        P_estimated = list()
        for r in reference_points:
            point = np.array(self.model.predict(np.matrix(r)).flatten())
            if point[0] > 0 and point[1] > 0:
                P_estimated.append(point)

        P_estimated = np.array(P_estimated)
        print("Estimado", P_estimated)
        Pt.decisionVariables = P_estimated  
        self.problem.evaluate(Pt)
        
        return Pt, Pt.objectives

    # DLV HyperVolume Based Implementation
    def execute_dvl_hv(
        self,
        dataset_size: int,
        max_evaluation,
        sampling,
        hv,
        n_closest: int,
        epsilon: float,
    ):
        solutions = sampling.random(n=dataset_size)
        objectives = np.array([self.problem.evaluate(sol)
                              for sol in solutions])
        reference_points = np.asarray(
            self.generate_reference_points()
        )

        P = set({s for s in solutions})
        P_best = set()
        HV_best = 0.0 # not infty, but me little float("inf")
        HV_prev = HV_best
        for _ in range(max_evaluation):
            P_rp = set()
            for r in reference_points:
                #assert len( self.models) == self.problem.numberOfDecisionVariables

                P_near, objectives_near = self.find_closest_solutions(
                    population=solutions, objectives=objectives, n_closest=n_closest
                )

                self.model.train(P_near, objectives_near)
                s = np.array(self.model.predict(np.matrix(r)))
                P_rp.add(s)

            HV_curr = hv(P_rp)
            # Lá ta  hv > best_hv:
            if HV_curr > HV_best:
                HV_best = HV_curr
                P_best = P_rp

            P.union(P_rp)
            # Why test for 0.0 ? 
            if HV_best != 0.0 and abs(HV_best - HV_prev) < epsilon:
                break
            HV_prev = HV_curr
        return P_best


    def execute_moea(self, population, objectives, max_evaluation, d):
        crossover = SBXCrossover(20.0, 0.9)
        mutation_probability = 0.5 #1.0 / self.problem.numberOfDecisionVariables
        mutation = PolynomialMutation(mutation_probability, 20.0)
        selection = BinaryTournament(2)
        self.moea = self.ClassMoea(
                problem=self.problem,
                maxEvaluations=max_evaluation,
                populationSize=self.pop_size,
                offSpringPopulationSize=self.pop_size,
                crossover=crossover,
                mutation=mutation,
                selection=selection,
                sparsity=CrowdingDistance(),
            )
        pf = ParetoFront()
        
        self.moea.execute(population)
        Pk = self.moea.population
        fronts = pf.fastNonDominatedSort(Pk)
        Pk.filter(fronts == 0)
        
        return Pk


    def find_closest_solutions(
        self, reference_point, population, objectives, n_closest: int
    ):
        rp = reference_point
        # Distance from current reference points stored in axis 1
        dist = np.linalg.norm(objectives - rp, axis=1)

        num_sol = population.shape[0]
        num_var = population.shape[1]
        num_obj = objectives.shape[1]

        obj_aux = np.zeros((num_sol, num_obj + 1))
        obj_aux[:, :-1] = objectives
        obj_aux[:, num_obj] = dist

        obj_ordenado = obj_aux[obj_aux[:, num_obj].argsort()]

        closest_objectives = np.zeros((n_closest, num_obj))
        closest_objectives = obj_ordenado[:n_closest, :-1]

        sol_aux = np.zeros((num_sol, num_var + 1))
        sol_aux[:, :-1] = population
        sol_aux[:, num_var] = dist

        sol_ordenado = sol_aux[sol_aux[:, num_var].argsort()]

        closest_solutions = np.zeros((n_closest, num_var))
        closest_solutions = sol_ordenado[:n_closest, :-1]
        return closest_solutions, closest_objectives


@lru_cache(maxsize=32)
def generate_reference_points(n_obj, n_div_per_obj=3):
    num_objs = n_obj
    num_divisions_per_obj = n_div_per_obj

    # Generates reference points for NSGA-III selection. This code is based on
    # jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>
    def gen(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left / total
            ref = deepcopy(work_point)
            return [ref]
        res = []
        for i in range(left + 1):
            work_point[depth] = i / total
            res = res + gen(
                work_point, num_objs, left - i, total, depth + 1
            )
        return res

    return gen(
        [0] * num_objs, num_objs, num_divisions_per_obj, num_divisions_per_obj, 0
    )


def save_obj_space(filename ,objectives, reference_points):
    xs = objectives[:, 0]
    ys = objectives[:, 1]
    zs = objectives[:, 2]

    #rxs = reference_points[:, 0]
    #rys = reference_points[:, 1]
    #rzs = reference_points[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(xs, ys, zs, color="blue")
    #ax.scatter3D(rxs, rys, rzs, color="orange")

    ax.set_xlabel("Objective X")
    ax.set_ylabel("Objective Y")
    ax.set_zlabel("Objective Z")

    plt.savefig(filename)
    plt.show()
