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

class BudgetExceededException(Exception):
    pass


class PyDOELatinHypercube:
    """Amostrador LHS baseado em pyDOE.lhs, reproduzindo fielmente o amostrador
    usado pelo Artur (dvl_framework.py usa `from pyDOE import *` e `lhs(...)`).

    O modelo inverso MLP (DTLZ2/DTLZ4) é extremamente sensível ao conjunto de
    treino; o amostrador LatinHypercube do scipy gera uma sequência diferente e
    produz hipervolumes bem menores no caso de DVL puro (e=250). O pyDOE
    reproduz os valores das Tabelas 19/20. Mantém a interface .random(n) do
    QMCEngine do scipy para não alterar execute_dvl.
    """

    def __init__(self, d, seed=None):
        self.d = d
        self.seed = seed

    def random(self, n):
        from pyDOE import lhs
        if self.seed is not None:
            np.random.seed(self.seed)
        return lhs(self.d, samples=n)

class SurrogateProblem(Problem):
    def __init__(self, real_problem: Problem, surrogate_models, real_eval_limit: int, reeval_interval: int | None = None, reeval_n: int | None = None):
        super(SurrogateProblem, self).__init__(
            numberOfObjectives=real_problem.numberOfObjectives,
            numberOfDecisionVariables=real_problem.numberOfDecisionVariables,
            decisionVariablesLimit=real_problem.decisionVariablesLimit
        )
        self.real_problem = real_problem
        self.surrogate_models = surrogate_models
        
        self.real_evals_count = 0
        self.real_eval_limit = real_eval_limit
        
        self.reeval_interval = reeval_interval
        self.reeval_n = reeval_n
        
        self.current_batch = []
        
        self.db_x = []
        self.db_y = []
        self.surrogate_training_time = 0.0

    def add_to_database(self, decision_variables, objectives):
        self.db_x.append(list(decision_variables))
        self.db_y.append(list(objectives))

    def retrain_surrogate(self):
        if len(self.db_x) > 0:
            import time
            start = time.perf_counter()
            X = np.array(self.db_x)
            Y = np.array(self.db_y)
            for i in range(self.numberOfObjectives):
                self.surrogate_models[i].fit(X, Y[:, i])
            self.surrogate_training_time += time.perf_counter() - start

    def evaluate(self, solution: Solution) -> Solution:
        if self.real_evals_count >= self.real_eval_limit:
            raise BudgetExceededException("Real evaluations budget exhausted")

        pred_objs = []
        X_test = np.array([solution.decisionVariables])
        for i in range(self.numberOfObjectives):
            pred_objs.append(self.surrogate_models[i].predict(X_test)[0])
            
        solution.objectives = pred_objs
        solution.evaluated = True
        
        self.current_batch.append(solution)
        
        interval = self.reeval_interval
        if interval is None:
            interval = 100
            
        if len(self.current_batch) >= interval:
            self.trigger_reevaluation()

        return solution

    def trigger_reevaluation(self):
        if not self.current_batch:
            return
            
        n_select = self.reeval_n
        if n_select is None or n_select >= len(self.current_batch):
            to_reeval = list(self.current_batch)
        else:
            to_reeval = self._select_best_candidates(self.current_batch, n_select)
            
        for sol in to_reeval:
            if self.real_evals_count >= self.real_eval_limit:
                break
                
            self.real_problem.evaluate(sol)
            sol.evaluated = True
            self.real_evals_count += 1
            
            self.add_to_database(sol.decisionVariables, sol.objectives)
            
        self.retrain_surrogate()
        self.current_batch.clear()

    def _select_best_candidates(self, candidates, n_select):
        if len(candidates) <= n_select:
            return candidates
        
        from src.ParetoFront import ParetoFront
        pf = ParetoFront()
        pf.fastNonDominatedSort(list(candidates))
        
        selected = []
        for front in pf.getInstance().front:
            if len(selected) + len(front) <= n_select:
                selected.extend(front)
            else:
                needed = n_select - len(selected)
                selected.extend(front[:needed])
                break
        return selected

    def evaluateConstraints(self, solution: Solution):
        return self.real_problem.evaluateConstraints(solution)

    def generateParetoFront(self):
        return self.real_problem.generateParetoFront()



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
            max_estimated_points: int | None = None,
            moea_population_size: int | None = None,
            max_training_samples: int | None = None,
            seed_moea_with_samples: bool = False,
            run_moea: bool = True,
            surrogate_class=None,
            surrogate_reeval_interval: int | None = None,
            surrogate_reeval_n: int | None = None,
            surrogate_max_evaluations: int = 100000,
            crossover=None,
            mutation=None,
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
        # Limita quantos pontos de referencia o DVL de fato ESTIMA (uma avaliacao
        # real por ponto). Permite desacoplar o custo de estimacao do DVL do
        # numero total de pontos de referencia do MOEA, de modo que o DVL respeite
        # um orcamento de avaliacoes (ver DVLMOEAWrapper). None = estima todos.
        self.max_estimated_points = max_estimated_points
        # Tamanho da populacao do MOEA na fase pos-DVL. Quando o numero de pontos
        # estimados pelo DVL e desacoplado (menor que os pontos de referencia), a
        # populacao estimada apenas SEMEIA o MOEA; o MOEA deve rodar com seu
        # tamanho natural (= numero de pontos de referencia), nao com o tamanho do
        # seed. None preserva o comportamento antigo (len(populacao inicial)).
        self.moea_population_size = moea_population_size
        # Teto de amostras usadas para TREINAR o modelo inverso por ponto de
        # referencia. O modelo satura bem antes de milhares de pontos; treinar em
        # um subconjunto fixo (selecionado uma vez) corta o custo dominante (o
        # treino e ~99% do tempo) sem perder qualidade da frente estimada.
        # None = treina em toda a amostra (comportamento antigo preservado).
        self.max_training_samples = max_training_samples
        # Quando True, semeia o MOEA com as MELHORES solucoes de (estimadas +
        # amostra LHS), nao so as estimadas. Como todas ja vem avaliadas
        # (evaluated=True), o MOEA nao as re-avalia -> custo zero de orcamento.
        # Evita "desperdicar" as amostras avaliadas que nao entraram no treino.
        self.seed_moea_with_samples = seed_moea_with_samples
        self.run_moea = run_moea
        self.surrogate_class = surrogate_class
        self.surrogate_reeval_interval = surrogate_reeval_interval
        self.surrogate_reeval_n = surrogate_reeval_n
        self.surrogate_max_evaluations = surrogate_max_evaluations
        self.crossover = crossover
        self.mutation = mutation
        """@obs
            API needed to be adapted because we can't determine the max eval upfront.
            we need to pass the moea class and construct it on the fly, to delay the 
            knowledge of max_eval value; 
        """
        self.ClassMoea = ClassMoea
        self.moea_kwargs = moea_kwargs
        self.model_training_time = 0.0

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
        self.model_training_time = 0.0
        sampling = PyDOELatinHypercube(
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
        if self.seed_moea_with_samples:
            # Semeia o MOEA com as melhores de (estimadas + amostra). Todas ja vem
            # avaliadas, entao isso nao gasta orcamento (os MOEAs pulam solucoes
            # com evaluated=True). Trunca pelo tamanho de populacao do MOEA.
            seed_size = (
                self.moea_population_size
                if self.moea_population_size is not None
                else len(trace["estimated_population_solutions"])
            )
            seed_pool = (
                list(trace["estimated_population_solutions"])
                + list(trace["sample_population_solutions"])
            )
            initial_population = select_best_solutions(seed_pool, seed_size)
        else:
            initial_population = trace["estimated_population_solutions"]
        estimated_front = trace["estimated_front_solutions"]

        original_problem = self.problem
        surrogate_problem = None

        if self.surrogate_class is not None and remaining_evaluations > 0:
            real_solutions = trace["sample_population_solutions"] + trace["estimated_population_solutions"]
            X_train = np.array([sol.decisionVariables for sol in real_solutions])
            Y_train = np.array([sol.objectives for sol in real_solutions])

            # Determine RandomForest parameter criterion based on version
            import sklearn
            criterion = "mse"
            try:
                from packaging.version import parse as parse_version
                if parse_version(sklearn.__version__) >= parse_version("1.2"):
                    criterion = "squared_error"
            except Exception:
                try:
                    parts = [int(p) for p in sklearn.__version__.split(".") if p.isdigit()]
                    if len(parts) >= 2 and (parts[0] > 1 or (parts[0] == 1 and parts[1] >= 2)):
                        criterion = "squared_error"
                except Exception:
                    pass

            import inspect
            sig = inspect.signature(self.surrogate_class)
            rf_kwargs = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 0,
                "warm_start": False,
                "criterion": criterion
            }
            if "n_jobs" in sig.parameters:
                rf_kwargs["n_jobs"] = -1

            surrogate_models = []
            for _ in range(self.problem.numberOfObjectives):
                model_inst = self.surrogate_class(**rf_kwargs)
                surrogate_models.append(model_inst)

            import time
            start_surr = time.perf_counter()
            for i in range(self.problem.numberOfObjectives):
                surrogate_models[i].fit(X_train, Y_train[:, i])
            self.model_training_time += time.perf_counter() - start_surr

            reeval_interval = self.surrogate_reeval_interval
            if reeval_interval is None:
                reeval_interval = len(initial_population)

            surrogate_problem = SurrogateProblem(
                real_problem=self.problem,
                surrogate_models=surrogate_models,
                real_eval_limit=remaining_evaluations,
                reeval_interval=reeval_interval,
                reeval_n=self.surrogate_reeval_n
            )
            for sol in real_solutions:
                surrogate_problem.add_to_database(sol.decisionVariables, sol.objectives)

            self.problem = surrogate_problem

        if self.run_moea and remaining_evaluations > 0:
            if surrogate_problem is not None:
                population = self.execute_moea(
                    initial_population=initial_population,
                    remaining_evaluations=self.surrogate_max_evaluations,
                )
                surrogate_problem.trigger_reevaluation()
            else:
                population = self.execute_moea(
                    initial_population=initial_population,
                    remaining_evaluations=remaining_evaluations,
                )
        else:
            population = {solution.clone() for solution in initial_population}

        self.problem = original_problem

        if surrogate_problem is not None:
            db_x_tuples = {tuple(x) for x in surrogate_problem.db_x}
            for sol in population:
                sol_tuple = tuple(sol.decisionVariables)
                if sol_tuple not in db_x_tuples:
                    if surrogate_problem.real_evals_count < surrogate_problem.real_eval_limit:
                        self.problem.evaluate(sol)
                        sol.evaluated = True
                        surrogate_problem.real_evals_count += 1
                        surrogate_problem.add_to_database(sol.decisionVariables, sol.objectives)

        final_front = extract_nondominated_solutions(population)
        
        actual_moea_real_evals = surrogate_problem.real_evals_count if surrogate_problem is not None else (self.moea.evaluations if self.moea else 0)
        total_real_evals_consumed = dvl_evaluations + actual_moea_real_evals

        if surrogate_problem is not None:
            self.model_training_time += surrogate_problem.surrogate_training_time
        
        trace.update({
            "model_training_time": self.model_training_time,
            "real_evaluation_time": self.problem.evaluation_time,
            "objective_calls": self.problem.avaliations,
            "dataset_size": dataset_size,
            "dvl_evaluations": dvl_evaluations,
            "remaining_evaluations": max(0, self.max_eval - total_real_evals_consumed),
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

        # Se ha um limite de pontos a estimar (orcamento do DVL), seleciona um
        # subconjunto espalhado uniformemente sobre o conjunto completo de pontos
        # de referencia, preservando a diversidade da frente estimada.
        if (
            self.max_estimated_points is not None
            and 0 < self.max_estimated_points < len(reference_points)
        ):
            idx = np.linspace(0, len(reference_points) - 1, self.max_estimated_points)
            idx = np.unique(np.round(idx).astype(int))
            reference_points = reference_points[idx]

        """@obs 
            It seems that in the code we use only 1 (one) model
            in case it's a list we use
        for M in self.models:
            # Each model is gi(Y) = xi or we can make one model g(Y) = X, correct?
            M.train(solutions, objectives)
        """
        # Estimation for a population as close as possible to the Pareto-optimal front.
        # O modelo é RE-TREINADO para cada ponto de referência, exatamente como o
        # Artur faz em fit_pred_new_solutions (o fit fica DENTRO do loop). Para
        # modelos estocásticos (MLP com random_state=None) cada re-treino gera
        # pesos diferentes, produzindo previsões diversas e bom espalhamento da
        # frente estimada — essencial para reproduzir DTLZ2/DTLZ4. Para modelos
        # determinísticos (SVR) o re-treino apenas repete o mesmo ajuste.
        # Subconjunto de treino: se max_training_samples limita o tamanho, escolhe
        # um subconjunto aleatorio FIXO (uma vez, com o RNG ja semeado por seed) e
        # treina nele em todas as iteracoes. A diversidade da frente vem do re-init
        # estocastico do MLP (random_state=None), nao dos dados, entao reduzir os
        # dados nao reduz o espalhamento — apenas o custo (e o lbfgs ate converge
        # melhor com menos pontos).
        if (
            self.max_training_samples is not None
            and 0 < self.max_training_samples < len(solutions)
        ):
            train_idx = np.random.choice(
                len(solutions), self.max_training_samples, replace=False
            )
            train_solutions = solutions[train_idx]
            train_objectives = transformed_objectives[train_idx]
        else:
            train_solutions = solutions
            train_objectives = transformed_objectives

        import time
        raw_predictions = list()
        bounded_predictions = list()
        for r in reference_points:
            start_train = time.perf_counter()
            self.model.train(train_solutions, train_objectives)
            self.model_training_time += time.perf_counter() - start_train

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
            "training_set_size": len(train_solutions),
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
        crossover = self.crossover if self.crossover is not None else SBXCrossover(30.0, 1.0)
        if self.mutation is not None:
            mutation = self.mutation
        else:
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
        moea_pop_size = (
            self.moea_population_size
            if self.moea_population_size is not None
            else len(initial_population)
        )
        if "populationSize" in params:
            kwargs["populationSize"] = moea_pop_size
        if "offSpringPopulationSize" in params:
            kwargs["offSpringPopulationSize"] = moea_pop_size
        if "numberOfDivisions" in params and self.reference_point_divisions is not None:
            kwargs["numberOfDivisions"] = self.reference_point_divisions

        # Merge self.moea_kwargs, overriding or adding extra parameters
        for k, v in self.moea_kwargs.items():
            if k in params:
                kwargs[k] = v

        self.moea = self.ClassMoea(**kwargs)
        try:
            res = self.moea.execute(
                initialPopulation={solution.clone() for solution in initial_population}
            )
            if res is not None:
                return res
            return self.moea.population
        except BudgetExceededException:
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


def select_best_solutions(population, k):
    """Seleciona as k melhores solucoes por ordenacao nao-dominada (fronts em
    ordem; o ultimo front incompleto e truncado). Usada para semear o MOEA com as
    melhores solucoes reais ja avaliadas (estimadas + amostra LHS), evitando a
    truncagem arbitraria que o MOEA/D faz quando recebe uma populacao maior."""
    population = [solution.clone() for solution in population]
    if len(population) <= k:
        return population

    pareto_front = ParetoFront()
    pareto_front.fastNonDominatedSort(population)
    selected = list()
    for front in pareto_front.getInstance().front:
        if len(selected) + len(front) <= k:
            selected.extend(front)
        else:
            selected.extend(front[: k - len(selected)])
            break
    return selected


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
