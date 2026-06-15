# Importando os Problemas
from src.problems.DTLZ import DTLZ1, DTLZ2, DTLZ3, DTLZ4
# Importando os Algoritmos
from src.MOEAs.MOEAD import MOEAD
from src.MOEAs.NSGAII import NSGAII
from src.MOEAs.NSGAIII import NSGAIII

from src.dvl.DVL import DVLFramework
from src.dvl.models.Linear import LinearModel

from utils.Time import measure_cpu_time
import numpy as np
import random
from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation
from src.MOEAs.crossovers.SBXCrossover import SBXCrossover
from src.BinaryTournament import BinaryTournament
from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance


def run_with_dvl(
    problem_class,      # Classe do problema
    moea_class,         # Classe do algoritmo evolucionário
    model_class,        # Classe do modelo de aprendizado de máquina
    m,                  # Número de objetivos do problema (M)
    k,                  # Parâmetro k do problema DTLZ
    max_evaluations,    # Número máximo de avaliações
    sample_size,        # Tamanho da amostra inicial (LHS) para treinar o DVL
    seed=42,            # Seed de aleatoriedade
    run_moea=True       # Se True, executa a fase evolutiva após o DVL; se False, apenas o DVL
):  
    np.random.seed(seed)
    random.seed(seed)
    
    problem = problem_class(numberOfObjectives=m, k=k)
    
    model = model_class()
    
    div_dict = {2: 99, 3: 12, 10: 3}
    num_div = div_dict.get(m, 12)
    
    framework = DVLFramework(
        pop_size=sample_size,
        max_eval=max_evaluations,
        ClassMoea=moea_class,
        model=model,
        problem=problem,
        reference_point_divisions=num_div,
        sampling_seed=seed,
        objective_transform="direction",
        run_moea=run_moea
    )
    
    population = framework.execute()
    return population


def run_without_dvl(
    problem_class,  # Classe do problema (ex: DTLZ1, DTLZ2)
    moea_class,     # Classe do algoritmo evolucionário (ex: NSGAIII, NSGAII)
    m,              # Número de objetivos do problema (M)
    k,              # Parâmetro k do problema DTLZ
    max_evaluations,# Número máximo de avaliações (max evaluations)
    seed=42         # Seed de aleatoriedade
):
    import inspect
    from src.Util import ReferencePoint

    np.random.seed(seed)
    random.seed(seed)

    problem = problem_class(numberOfObjectives=m, k=k)
    
    div_dict = {2: 99, 3: 12, 10: 3}
    num_div = div_dict.get(m, 12)
    
    crossover = SBXCrossover(20.0, 0.9)
    mutation_probability = 1.0 / problem.numberOfDecisionVariables
    mutation = PolynomialMutation(mutation_probability, 20.0)
    selection = BinaryTournament()
    sparsity = CrowdingDistance()
    
    # Gera pontos de referência para deduzir tamanho de população se o algoritmo precisar
    ref_points = ReferencePoint().generateReferencePoints(m, num_div)
    pop_size = len(ref_points)

    # Identificar assinatura do construtor dinamicamente
    sig = inspect.signature(moea_class)
    params = sig.parameters
    
    kwargs = {}
    if "problem" in params:
        kwargs["problem"] = problem
    if "maxEvaluations" in params:
        kwargs["maxEvaluations"] = max_evaluations
    if "crossover" in params:
        kwargs["crossover"] = crossover
    if "mutation" in params:
        kwargs["mutation"] = mutation
    if "selection" in params:
        kwargs["selection"] = selection
    if "sparsity" in params:
        kwargs["sparsity"] = sparsity
    if "numberOfDivisions" in params:
        kwargs["numberOfDivisions"] = num_div
    if "populationSize" in params:
        kwargs["populationSize"] = pop_size
    if "offSpringPopulationSize" in params:
        kwargs["offSpringPopulationSize"] = pop_size if pop_size % 2 == 0 else pop_size + 1
        
    moea = moea_class(**kwargs)
    
    population = moea.execute()
    
    return population


def main():
    # Array de parâmetros fornecido para os experimentos
    parameters = [
        # m -> número de objetivos
        # problem_k -> parâmetro k do problema DTLZ (k = n - m + 1)
        # max_evaluations -> número máximo de avaliações
        # sample_size -> quantos pontos iniciais o DVL usa para aprender (usado como baseline)

        # Configuração com 3 Objetivos
        {"m": 3, "problem_k": 10, "max_evaluations": 250,   "sample_size": 159},
        {"m": 3, "problem_k": 10, "max_evaluations": 500,   "sample_size": 227},
        {"m": 3, "problem_k": 10, "max_evaluations": 1000,  "sample_size": 363},
        {"m": 3, "problem_k": 10, "max_evaluations": 1500,  "sample_size": 681},
        {"m": 3, "problem_k": 10, "max_evaluations": 10000, "sample_size": 8726},
        # Configuração com 10 Objetivos
        {"m": 10, "problem_k": 1,  "max_evaluations": 250,   "sample_size": 50},
        {"m": 10, "problem_k": 1,  "max_evaluations": 500,   "sample_size": 112},
        {"m": 10, "problem_k": 1,  "max_evaluations": 1000,  "sample_size": 132},
        {"m": 10, "problem_k": 1,  "max_evaluations": 1500,  "sample_size": 464},
        {"m": 10, "problem_k": 1,  "max_evaluations": 10000, "sample_size": 7388},
    ]
    
    problems = [DTLZ1, DTLZ2, DTLZ3, DTLZ4]
    algorithms = [NSGAIII]
    # dvl_fractions = [0.25, 0.50, 0.75]
    
    for param in parameters:
        m = param["m"]
        k = param["problem_k"]
        max_eval = param["max_evaluations"]
        sample_size_baseline = param["sample_size"]
        
        for problem_class in problems:
            for algo_class in algorithms:
                
                # 1. Experimento sem DVL (MOEA puro)
                label_nodvl = f"{algo_class.__name__} puro no {problem_class.__name__} (M={m}, E={max_eval})"
                with measure_cpu_time(label_nodvl):
                    run_without_dvl(problem_class, algo_class, m, k, max_eval)
                
                # 2. Experimento com DVL (Baseline do array) + MOEA
                label_dvl_base = f"DVL (Baseline LHS={sample_size_baseline}) + {algo_class.__name__} no {problem_class.__name__} (M={m}, E={max_eval})"
                with measure_cpu_time(label_dvl_base):
                    run_with_dvl(problem_class, algo_class, LinearModel, m, k, max_eval, sample_size_baseline, run_moea=True)

if __name__ == '__main__':
    main()