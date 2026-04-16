#!/usr/bin/env python3

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from src.QualityIndicator import HV, IGD
from src.MOEAs.NSGAIII import NSGAIII
from src.dvl.DVL import DVLFramework
from src.dvl.models.Linear import LinearModel
from src.dvl.models.MLP import MLPModel
from src.problems.DTLZ import DTLZ1
from utils.CSV import save_dvl_results_csvs


DEFAULT_PROBLEM_NAME = "DTLZ1"
DEFAULT_OBJECTIVES = 3
DEFAULT_PROBLEM_K = 10
DEFAULT_EVALUATIONS = 500
DEFAULT_MODEL_NAME = "linear"
DEFAULT_SEED = 42
DEFAULT_OUTPUT_ROOT = Path("out/dvl")
DEFAULT_OBJECTIVE_TRANSFORM = "direction"

DTLZ1_TABLE_18 = {
    3: {250: 159, 500: 227, 1000: 250, 1500: 300, 10000: 300},
    10: {250: 50, 500: 112, 1000: 200, 1500: 200, 10000: 300},
}

REFERENCE_POINT_DIVISIONS = {
    3: 12,
    10: 3,
}

HV_REFERENCE_POINTS = {
    ("DTLZ1", 3): np.array([1.0, 1.0, 1.0], dtype=float),
    ("DTLZ1", 10): np.array([5.0] * 10, dtype=float),
}


def get_problem_n(problem_k: int, m: int) -> int:
    return problem_k + m - 1


def get_sample_size(problem_name: str, m: int, e: int) -> int:
    if problem_name != "DTLZ1":
        raise ValueError(f"Problema nao suportado: {problem_name}")
    if m not in DTLZ1_TABLE_18 or e not in DTLZ1_TABLE_18[m]:
        raise ValueError(f"Configuracao nao encontrada na Tabela 18 para m={m} e e={e}")
    return DTLZ1_TABLE_18[m][e]


def get_reference_front_path(problem_name: str, m: int) -> Path:
    if problem_name != "DTLZ1":
        raise ValueError(f"Problema nao suportado: {problem_name}")
    if m == 3:
        return Path("resources/ReferenceFronts/DTLZ/DTLZ1.3D.csv")
    if m == 10:
        return Path("resources/ReferenceFronts/DTLZ/DTLZ1-pareto_10.txt")
    raise ValueError(f"Fronteira de referencia nao configurada para m={m}")


def load_reference_front(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float)
    return np.loadtxt(path, dtype=float)


def get_model(model_name: str, seed: int):
    if model_name == "linear":
        return LinearModel()
    if model_name == "mlp":
        return MLPModel((11, 11, 11), random_state=seed, max_iter=2000)
    raise ValueError(f"Modelo nao suportado: {model_name}")


def get_hv_reference_point(problem_name: str, m: int) -> np.ndarray:
    key = (problem_name, m)
    if key not in HV_REFERENCE_POINTS:
        raise ValueError(f"Ponto de referencia HV nao configurado para {problem_name} com m={m}")
    return HV_REFERENCE_POINTS[key]


def compute_igd(reference_front: np.ndarray, front: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return float("nan")
    return IGD(reference_front.tolist()).calculate(front.tolist())


def compute_hv(front: np.ndarray, ref_point: np.ndarray) -> tuple[float, int, int]:
    indicator = HV(referencePoint=ref_point)
    value = indicator.calculate(front)
    normalized_value = value / np.prod(ref_point)
    return normalized_value, indicator.last_valid_count, indicator.last_total_count


def build_run_dir(
    output_root: Path,
    problem_name: str,
    m: int,
    n: int,
    e: int,
    sample_size: int,
    model_name: str,
) -> Path:
    return output_root / f"{problem_name.lower()}_m{m}_n{n}_e{e}_k{sample_size}_{model_name}"


def run_experiment(
    *,
    problem_name: str = DEFAULT_PROBLEM_NAME,
    m: int = DEFAULT_OBJECTIVES,
    e: int = DEFAULT_EVALUATIONS,
    sample_size: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    seed: int = DEFAULT_SEED,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    objective_transform: str | None = DEFAULT_OBJECTIVE_TRANSFORM,
    problem_k: int = DEFAULT_PROBLEM_K,
) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    if problem_name != "DTLZ1":
        raise ValueError(f"Problema nao suportado: {problem_name}")
    if m not in REFERENCE_POINT_DIVISIONS:
        raise ValueError(f"Numero de objetivos nao suportado para este runner: m={m}")

    if sample_size is None:
        sample_size = get_sample_size(problem_name, m, e)

    n = get_problem_n(problem_k, m)
    output_root = Path(output_root)
    reference_front = load_reference_front(get_reference_front_path(problem_name, m))

    framework = DVLFramework(
        pop_size=sample_size,
        max_eval=e,
        ClassMoea=NSGAIII,
        model=get_model(model_name, seed),
        problem=DTLZ1(numberOfObjectives=m, k=problem_k),
        numberOfDivisions=REFERENCE_POINT_DIVISIONS[m],
        sampling_seed=seed,
        objective_transform=objective_transform,
    )

    _, trace = framework.execute(return_trace=True)

    estimated_igd = compute_igd(reference_front, trace["estimated_objectives"])
    final_igd = compute_igd(reference_front, trace["final_front_objectives"])
    final_hv, hv_valid_count, hv_total_count = compute_hv(
        trace["final_front_objectives"],
        get_hv_reference_point(problem_name, m),
    )

    run_dir = build_run_dir(output_root, problem_name, m, n, e, sample_size, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    legacy_csv = run_dir / "dvl_results.csv"
    if legacy_csv.exists():
        legacy_csv.unlink()

    csv_dir = save_dvl_results_csvs(run_dir, trace, reference_front, m, n)

    return {
        "problem_name": problem_name,
        "objectives": m,
        "problem_k": problem_k,
        "decision_variables": n,
        "evaluations": e,
        "sample_size": sample_size,
        "model_name": model_name,
        "seed": seed,
        "objective_transform": objective_transform,
        "run_dir": run_dir,
        "csv_dir": csv_dir,
        "estimated_igd": estimated_igd,
        "final_igd": final_igd,
        "final_hv": final_hv,
        "hv_valid_count": hv_valid_count,
        "hv_total_count": hv_total_count,
        "dataset_size": trace["dataset_size"],
        "dvl_evaluations": trace["dvl_evaluations"],
        "remaining_evaluations": trace["remaining_evaluations"],
        "final_front_size": len(trace["final_front_objectives"]),
    }
