import numpy as np
from functools import lru_cache

from copy import deepcopy
from scipy.stats.qmc import LatinHypercube, QMCEngine
from src.Solution import Solution
from src.BinaryTournament import BinaryTournament
from src.MOEAs.crossovers.SBXCrossover import SBXCrossover
from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation
from src.ParetoFront import ParetoFront

from src.problems.Problem import Problem
from src.dvl.Model import Model


class DVLFramework:
    # for a problem with 'n' variables, '1' supervised learning models will be used
    def __init__(self,
            pop_size:int, # k, initial popsize
            max_eval:int, # e, max number of objective evaluation 
            ClassMoea, 
            model: Model, 
            problem: Problem,
            reference_point_divisions: int | None = None,
            reference_points=None,
            sampling_seed: int | None = None,
            clip_decision_variables: bool = True,
            objective_transform: str | None = None,
            training_evaluations: int | None = None,
            run_moea: bool = True,
            **moea_kwargs
        ):
        self.pop_size = pop_size
        self.max_eval = max_eval
        self.problem: Problem = problem
        self.model: Model = model
        self.moea = None
        self.reference_points = reference_points
        self.reference_point_divisions = reference_point_divisions
        self.sampling_seed = sampling_seed
        self.clip_decision_variables = clip_decision_variables
        self.objective_transform = objective_transform
        self.training_evaluations = training_evaluations
        self.run_moea = run_moea
        """@obs
            API needed to be adapted because we can't determine the max eval upfront.
            we need to pass the moea class and construct it on the fly, to delay the 
            knowledge of max_eval value; 
        """
        self.ClassMoea = ClassMoea
        self.moea_kwargs = moea_kwargs

    def _decision_variable_bounds(self):
        lower = np.asarray(self.problem.decisionVariablesLimit[0], dtype=float)
        upper = np.asarray(self.problem.decisionVariablesLimit[1], dtype=float)
        return lower, upper

    def _clip_decision_vector(self, decision_variables):
        vector = np.asarray(decision_variables, dtype=float).reshape(-1)
        if not self.clip_decision_variables:
            return vector

        lower, upper = self._decision_variable_bounds()
        return np.clip(vector, lower, upper)

    def _evaluate_decision_vector(self, decision_variables):
        bounded_variables = self._clip_decision_vector(decision_variables)
        solution = Solution(
            self.problem.numberOfObjectives,
            self.problem.numberOfDecisionVariables,
            bounded_variables,
        )
        evaluated = self.problem.evaluate(solution)
        evaluated.evaluated = True
        return evaluated

    def _transform_objective_vectors(self, objective_vectors):
        objective_vectors = np.asarray(objective_vectors, dtype=float)
        if self.objective_transform is None:
            return objective_vectors

        if self.objective_transform == "direction":
            sums = np.sum(objective_vectors, axis=1, keepdims=True)
            safe_sums = np.where(np.abs(sums) > 1e-12, sums, 1.0)
            return objective_vectors / safe_sums

        raise ValueError(f"Transformacao de objetivos nao suportada: {self.objective_transform}")

    def _transform_reference_point(self, reference_point):
        reference_point = np.asarray(reference_point, dtype=float).reshape(1, -1)
        return self._transform_objective_vectors(reference_point).reshape(-1)

    def _get_reference_points(self, reference_points=None):
        if reference_points is not None:
            return np.asarray(reference_points, dtype=float)

        if self.reference_points is not None:
            return np.asarray(self.reference_points, dtype=float)

        n_divisions = self.reference_point_divisions
        if n_divisions is None:
            n_divisions = self.moea_kwargs.get("numberOfDivisions", 2)

        return np.asarray(
            generate_reference_points(
                n_obj=self.problem.numberOfObjectives,
                n_div_per_obj=n_divisions,
            ),
            dtype=float,
        )

    def execute(self, return_trace: bool = False):
        sampling: QMCEngine = LatinHypercube(
            d=self.problem.numberOfDecisionVariables,
            seed=self.sampling_seed,
        )
        dataset_size = self.training_evaluations if self.training_evaluations is not None else self.pop_size
        estimated_population, _, trace = self.execute_dvl(
            sampling=sampling,
            dataset_size=dataset_size,
            return_trace=True,
        )

        dvl_evaluations = dataset_size + len(estimated_population)
        remaining_evaluations = max(0, self.max_eval - dvl_evaluations)
        initial_population = trace["estimated_population_solutions"]
        estimated_front = trace["estimated_front_solutions"]

        if self.run_moea and remaining_evaluations > 0:
            population = self.execute_moea(
                initial_population=initial_population,
                remaining_evaluations=remaining_evaluations,
            )
        else:
            population = {solution.clone() for solution in initial_population}

        final_front = extract_nondominated_solutions(population)
        trace.update({
            "dataset_size": dataset_size,
            "dvl_evaluations": dvl_evaluations,
            "remaining_evaluations": remaining_evaluations,
            "estimated_front_solutions": estimated_front,
            "estimated_front_objectives": solutions_to_objectives(
                estimated_front,
                self.problem.numberOfObjectives,
            ),
            "estimated_front_decision_vectors": solutions_to_decision_vectors(
                estimated_front,
                self.problem.numberOfDecisionVariables,
            ),
            "final_population": list(population),
            "final_objectives": solutions_to_objectives(
                population,
                self.problem.numberOfObjectives,
            ),
            "final_decision_vectors": solutions_to_decision_vectors(
                population,
                self.problem.numberOfDecisionVariables,
            ),
            "final_front_solutions": final_front,
            "final_front_objectives": solutions_to_objectives(
                final_front,
                self.problem.numberOfObjectives,
            ),
            "final_front_decision_vectors": solutions_to_decision_vectors(
                final_front,
                self.problem.numberOfDecisionVariables,
            ),
        })

        if return_trace:
            return population, trace
        return population

    # DVL Inverse Modeling only
    # pg. 80 "The necessity of using the hypervolume inside the algorithm is eliminated"
    def execute_dvl(
        self,
        sampling: QMCEngine,
        dataset_size: int,
        reference_points=None,
        return_trace: bool = False,
    ):
        solutions = sampling.random(n=dataset_size)

        """@obs
            .evaluate doesnt take a array as input
            also it doesnt return objetives, it returns the same solution 
            from which i should extract the objetives on the from the variable
            .objectives 

            class Solution() needed to be changed because we can't pass a already
            valid solution as np.array to the constructor 
        """
        sampled_population = [
            self._evaluate_decision_vector(solution)
            for solution in solutions
        ]
        objectives = solutions_to_objectives(sampled_population)
        transformed_objectives = self._transform_objective_vectors(objectives)
        reference_points = self._get_reference_points(reference_points)
        
        """@obs 
            It seems that in the code we use only 1 (one) model
            in case it's a list we use
        for M in self.models:
            # Each model is gi(Y) = xi or we can make one model g(Y) = X, correct?
            M.train(solutions, objectives)
        """
        # ONE model is g(Y) = X, correct?
        self.model.train(solutions, transformed_objectives)

        # Estimation for a population as close as possible to the Pareto-optimal front
        raw_predictions = list()
        bounded_predictions = list()
        for r in reference_points:
            prediction = self.model.predict(self._transform_reference_point(r))
            raw_prediction = np.asarray(prediction, dtype=float).reshape(-1)
            raw_predictions.append(raw_prediction)
            bounded_predictions.append(self._clip_decision_vector(raw_prediction))

        estimated_population = [
            self._evaluate_decision_vector(solution)
            for solution in bounded_predictions
        ]
        estimated_front = extract_nondominated_solutions(estimated_population)
        estimated_objectives = solutions_to_objectives(
            estimated_population,
            self.problem.numberOfObjectives,
        )
        raw_predictions_np = np.asarray(raw_predictions, dtype=float)
        bounded_predictions_np = np.asarray(bounded_predictions, dtype=float)
        lower_bounds, upper_bounds = self._decision_variable_bounds()
        out_of_bounds_mask = (raw_predictions_np < lower_bounds) | (raw_predictions_np > upper_bounds)

        trace = {
            "sample_decision_vectors": np.asarray(solutions, dtype=float),
            "sample_objectives": objectives,
            "sample_population_solutions": sampled_population,
            "reference_points": reference_points,
            "transformed_sample_objectives": transformed_objectives,
            "transformed_reference_points": np.asarray(
                [self._transform_reference_point(r) for r in reference_points],
                dtype=float,
            ),
            "estimated_decision_vectors_raw": raw_predictions_np,
            "estimated_decision_vectors_bounded": bounded_predictions_np,
            "estimated_population_solutions": estimated_population,
            "estimated_objectives": estimated_objectives,
            "estimated_front_solutions": estimated_front,
            "estimated_front_objectives": solutions_to_objectives(
                estimated_front,
                self.problem.numberOfObjectives,
            ),
            "predicted_out_of_bounds_mask": out_of_bounds_mask,
            "predicted_out_of_bounds_count": int(np.count_nonzero(out_of_bounds_mask)),
            "predicted_out_of_bounds_solutions": int(
                np.count_nonzero(np.any(out_of_bounds_mask, axis=1))
            ),
            "decision_variable_lower_bounds": lower_bounds,
            "decision_variable_upper_bounds": upper_bounds,
        }

        if return_trace:
            return bounded_predictions, estimated_objectives, trace
        return bounded_predictions, estimated_objectives

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
        objectives = np.array([
            self._evaluate_decision_vector(sol).objectives
            for sol in solutions
        ])
        reference_points = np.asarray(
            generate_reference_points(n_obj=self.problem.numberOfObjectives)
        )

        P = set({s for s in solutions})
        P_best = set()
        HV_best = 0.0 # not infty, but me little float("inf")
        HV_prev = HV_best
        for _ in range(max_evaluation):
            P_rp = set()
            for r in reference_points:
                # assert len( self.models) == self.problem.numberOfDecisionVariables

                P_near, objectives_near = self.find_closest_solutions(
                    reference_point=r,
                    population=solutions,
                    objectives=objectives,
                    n_closest=n_closest,
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


    def execute_moea(self, initial_population, remaining_evaluations):
        crossover = SBXCrossover(20.0, 0.9)
        mutation_probability = 1.0 / self.problem.numberOfDecisionVariables
        mutation = PolynomialMutation(mutation_probability, 20.0)
        selection = BinaryTournament()

        try:
            from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance
            sparsity = CrowdingDistance()
        except ImportError:
            sparsity = None

        import inspect
        sig = inspect.signature(self.ClassMoea)
        params = sig.parameters

        kwargs = {}
        if "problem" in params:
            kwargs["problem"] = self.problem
        if "maxEvaluations" in params:
            kwargs["maxEvaluations"] = remaining_evaluations
        if "crossover" in params:
            kwargs["crossover"] = crossover
        if "mutation" in params:
            kwargs["mutation"] = mutation
        if "selection" in params:
            kwargs["selection"] = selection
        if "sparsity" in params:
            kwargs["sparsity"] = sparsity
        if "populationSize" in params:
            kwargs["populationSize"] = len(initial_population)
        if "offSpringPopulationSize" in params:
            kwargs["offSpringPopulationSize"] = len(initial_population)

        # Merge self.moea_kwargs, overriding or adding extra parameters
        for k, v in self.moea_kwargs.items():
            if k in params:
                kwargs[k] = v

        self.moea = self.ClassMoea(**kwargs)
        res = self.moea.execute(
            initialPopulation={solution.clone() for solution in initial_population}
        )
        if res is not None:
            return res
        return self.moea.population


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


def extract_nondominated_solutions(population):
    population = [solution.clone() for solution in population]
    if not population:
        return list()

    pareto_front = ParetoFront()
    pareto_front.fastNonDominatedSort(population)
    return pareto_front.getFront(0)


def solutions_to_objectives(population, number_of_objectives=None):
    objectives = [solution.objectives for solution in population]
    if not objectives:
        width = 0 if number_of_objectives is None else number_of_objectives
        return np.empty((0, width), dtype=float)

    return np.asarray(objectives, dtype=float)


def solutions_to_decision_vectors(population, number_of_variables=None):
    decision_vectors = [solution.decisionVariables for solution in population]
    if not decision_vectors:
        width = 0 if number_of_variables is None else number_of_variables
        return np.empty((0, width), dtype=float)

    return np.asarray(decision_vectors, dtype=float)


def save_obj_space(filename, objectives, reference_points, show=False):
    from matplotlib import pyplot as plt

    xs = objectives[:,0]
    ys = objectives[:,1]
    zs = objectives[:,2]

    rxs = reference_points[:,0]
    rys = reference_points[:,1]
    rzs = reference_points[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(xs, ys, zs, color="blue")
    ax.scatter3D(rxs, rys, rzs, color="orange")

    ax.set_xlabel("Objective X")
    ax.set_ylabel("Objective Y")
    ax.set_zlabel("Objective Z")

    plt.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)
