import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
from concurrent.futures import ProcessPoolExecutor
import os

from src.QualityIndicator import HV, IGD
from src.problems.DTLZ import DTLZ1
from src.MOEAs.NSGAII import NSGAII
from src.MOEAs.MOEAD import MOEAD
from src.MOEAs.NSGAIII import NSGAIII
from src.dvl.DVL import DVLFramework
from src.dvl.models.Linear import LinearModel
from src.dvl.models.MLP import MLPModel
from src.MOEAs.crossovers.SBXCrossover import SBXCrossover
from src.MOEAs.mutations.PolynomialMutation import PolynomialMutation
from src.BinaryTournament import BinaryTournament

REFERENCE_POINT_DIVISIONS = {
    3: 12,
    10: 3,
}

HV_REFERENCE_POINTS = {
    ("DTLZ1", 3): np.array([1.0, 1.0, 1.0], dtype=float),
    ("DTLZ1", 10): np.array([5.0] * 10, dtype=float),
}

def get_reference_front_path(problem_name: str, m: int) -> Path:
    if m == 3:
        return Path("resources/ReferenceFronts/DTLZ/DTLZ1.3D.csv")
    if m == 10:
        return Path("resources/ReferenceFronts/DTLZ/DTLZ1-pareto_10.txt")
    raise ValueError(f"Fronteira de referencia nao configurada para m={m}")

def load_reference_front(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float)
    return np.loadtxt(path, dtype=float)

def compute_igd(reference_front: np.ndarray, front: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return float("nan")
    return IGD(reference_front.tolist()).calculate(front.tolist())

def compute_hv(front: np.ndarray, ref_point: np.ndarray) -> float:
    indicator = HV(referencePoint=ref_point)
    value = indicator.calculate(front)
    normalized_value = float(value / np.prod(ref_point))
    return normalized_value

def extract_nondominated_solutions(population):
    from src.ParetoFront import ParetoFront
    population = [solution.clone() for solution in population]
    if not population:
        return list()
    pareto_front = ParetoFront()
    pareto_front.fastNonDominatedSort(population)
    return pareto_front.getFront(0)

def solutions_to_objectives(population):
    return np.asarray([solution.objectives for solution in population], dtype=float)

def run_standard_moea(ClassMoea, problem, max_evals, pop_size, seed):
    np.random.seed(seed)
    random.seed(seed)
    
    crossover = SBXCrossover(20.0, 0.9)
    mutation_probability = 1.0 / problem.numberOfDecisionVariables
    mutation = PolynomialMutation(mutation_probability, 20.0)
    selection = BinaryTournament()
    
    try:
        from src.MOEAs.sparsities.CrowdingDistance import CrowdingDistance
        sparsity = CrowdingDistance()
    except ImportError:
        sparsity = None
        
    moea = ClassMoea(
        problem=problem,
        maxEvaluations=max_evals,
        populationSize=pop_size,
        offSpringPopulationSize=pop_size,
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        sparsity=sparsity
    )
    
    t0_proc = time.process_time()
    moea.execute()
    t_proc = time.process_time() - t0_proc
    
    final_front = extract_nondominated_solutions(moea.population)
    front_objectives = solutions_to_objectives(final_front)
    
    return front_objectives, t_proc

def run_dvl_framework(ClassMoea, problem, max_evals, training_evals, run_moea, seed, model_name="linear"):
    np.random.seed(seed)
    random.seed(seed)
    
    m = problem.numberOfObjectives
    
    if model_name == "linear":
        model = LinearModel()
    elif model_name == "mlp":
        model = MLPModel((11, 11, 11), random_state=seed, max_iter=2000)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    divisions = REFERENCE_POINT_DIVISIONS[m]
    
    framework = DVLFramework(
        pop_size=training_evals,
        max_eval=max_evals,
        ClassMoea=ClassMoea,
        model=model,
        problem=problem,
        reference_point_divisions=divisions,
        sampling_seed=seed,
        objective_transform="direction",
        training_evaluations=training_evals,
        run_moea=run_moea
    )
    
    t0_proc = time.process_time()
    _, trace = framework.execute(return_trace=True)
    t_proc = time.process_time() - t0_proc
    
    front_objectives = trace["final_front_objectives"]
    return front_objectives, t_proc

def run_single_experiment(args):
    m, algorithm, mode, total_evals, training_evals, seed, model_name = args
    
    problem = DTLZ1(numberOfObjectives=m, k=10)
    ref_front = load_reference_front(get_reference_front_path("DTLZ1", m))
    ref_point = HV_REFERENCE_POINTS[("DTLZ1", m)]
    
    if m == 3:
        pop_size = 91
    elif m == 10:
        pop_size = 220
    else:
        pop_size = 100
        
    if mode == "surrogate_only":
        front, t_proc = run_dvl_framework(
            ClassMoea=NSGAIII,
            problem=problem,
            max_evals=training_evals + pop_size,
            training_evals=training_evals,
            run_moea=False,
            seed=seed,
            model_name=model_name
        )
        igd = compute_igd(ref_front, front)
        hv = compute_hv(front, ref_point)
        actual_evals = training_evals + len(front)
    elif mode == "outside_dvl":
        ClassMoea = NSGAII if algorithm == "NSGAII" else MOEAD
        front, t_proc = run_standard_moea(
            ClassMoea=ClassMoea,
            problem=problem,
            max_evals=total_evals,
            pop_size=pop_size,
            seed=seed
        )
        igd = compute_igd(ref_front, front)
        hv = compute_hv(front, ref_point)
        actual_evals = total_evals
    elif mode == "inside_dvl":
        ClassMoea = NSGAII if "NSGAII" in algorithm else MOEAD
        front, t_proc = run_dvl_framework(
            ClassMoea=ClassMoea,
            problem=problem,
            max_evals=total_evals,
            training_evals=training_evals,
            run_moea=True,
            seed=seed,
            model_name=model_name
        )
        igd = compute_igd(ref_front, front)
        hv = compute_hv(front, ref_point)
        actual_evals = total_evals
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return {
        "m": m,
        "algorithm": algorithm,
        "mode": mode,
        "total_evals": actual_evals,
        "training_evals": training_evals,
        "seed": seed,
        "proc_time": t_proc,
        "igd": igd,
        "hv": hv,
        "model": model_name
    }

def execute_experiments(m_list=[3], n_runs=30, max_evals_list=[10000, 100000], output_path="out/experiments_results.csv", n_workers=None):
    Path("out").mkdir(parents=True, exist_ok=True)
    configs = []
    
    seeds = [42 + i for i in range(n_runs)]
    
    for m in m_list:
        # DVL Isolado
        training_cases = [50, 112, 200, 300]
        for tr_eval in training_cases:
            for seed in seeds:
                configs.append((m, "DVL-Isolado", "surrogate_only", 0, tr_eval, seed, "linear"))
                
        # MOEAs fora e dentro do DVL
        for max_evals in max_evals_list:
            if m == 10 and max_evals > 10000:
                continue
                
            # Sem DVL
            for name in ["NSGAII", "MOEAD"]:
                for seed in seeds:
                    configs.append((m, name, "outside_dvl", max_evals, 0, seed, "none"))
                    
            # Com DVL
            tr_evals_choices = [300, 1000, 5000] if max_evals >= 10000 else [112, 200]
            for tr_eval in tr_evals_choices:
                if tr_eval >= max_evals:
                    continue
                for name in ["NSGAII", "MOEAD"]:
                    for seed in seeds:
                        configs.append((m, f"DVL+{name}", "inside_dvl", max_evals, tr_eval, seed, "linear"))
                        
    print(f"Total configurations to run: {len(configs)}")
    
    if n_workers is None:
        n_workers = os.cpu_count()
        
    print(f"Running in parallel using {n_workers} processes...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(run_single_experiment, configs))
        
    df = pd.DataFrame(results)
    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"All experiments completed! Results saved to {output_path}")
    else:
        print("All experiments completed!")
    return df

def analyze_and_compare(df_path="out/experiments_results.csv"):
    df = pd.read_csv(df_path)
    
    summary = df.groupby(["m", "algorithm", "total_evals", "training_evals"]).agg({
        "hv": ["mean", "std"],
        "igd": ["mean", "std"],
        "proc_time": ["mean", "std"]
    }).reset_index()
    
    summary.columns = [
        "Objectives", "Algorithm", "Total_Evals", "Training_Evals",
        "HV_mean", "HV_std", "IGD_mean", "IGD_std", "Time_mean", "Time_std"
    ]
    
    print("\n--- Summary Statistics ---")
    print(summary.to_string(index=False))
    print("\n--- Statistical Significance Tests (Mann-Whitney U, p-value < 0.05 is significant) ---")
    
    grouped = df.groupby(["m", "total_evals"])
    for (m, total_evals), group in grouped:
        if total_evals == 100000 or total_evals == 10000:
            print(f"\nFor m={m}, Budget={total_evals}:")
            # Compare NSGAII fora vs NSGAII dentro do DLV
            nsgaii_out = group[(group["algorithm"] == "NSGAII") & (group["mode"] == "outside_dvl")]["hv"].values
            moead_out = group[(group["algorithm"] == "MOEAD") & (group["mode"] == "outside_dvl")]["hv"].values
            
            dvl_nsgaii_groups = group[group["algorithm"] == "DVL+NSGAII"]
            for tr_eval in dvl_nsgaii_groups["training_evals"].unique():
                nsgaii_in = dvl_nsgaii_groups[dvl_nsgaii_groups["training_evals"] == tr_eval]["hv"].values
                if len(nsgaii_out) > 0 and len(nsgaii_in) > 0:
                    try:
                        stat, p_val = mannwhitneyu(nsgaii_out, nsgaii_in, alternative="two-sided")
                        print(f"  NSGAII (outside) vs DVL+NSGAII (inside, training={tr_eval}) HV p-value: {p_val:.4e} "
                              f"({'Significant' if p_val < 0.05 else 'Not Significant'})")
                    except Exception as e:
                        print(f"  NSGAII (outside) vs DVL+NSGAII (inside, training={tr_eval}) error: {e}")
                    
            dvl_moead_groups = group[group["algorithm"] == "DVL+MOEAD"]
            for tr_eval in dvl_moead_groups["training_evals"].unique():
                moead_in = dvl_moead_groups[dvl_moead_groups["training_evals"] == tr_eval]["hv"].values
                if len(moead_out) > 0 and len(moead_in) > 0:
                    try:
                        stat, p_val = mannwhitneyu(moead_out, moead_in, alternative="two-sided")
                        print(f"  MOEAD (outside) vs DVL+MOEAD (inside, training={tr_eval}) HV p-value: {p_val:.4e} "
                              f"({'Significant' if p_val < 0.05 else 'Not Significant'})")
                    except Exception as e:
                        print(f"  MOEAD (outside) vs DVL+MOEAD (inside, training={tr_eval}) error: {e}")
                          
    return summary

if __name__ == "__main__":
    execute_experiments(m_list=[3], n_runs=1, max_evals_list=[10000], output_path="out/test_pipeline_results.csv")
    analyze_and_compare("out/test_pipeline_results.csv")
