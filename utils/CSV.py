import csv
import os

import numpy as np


def save_to_csv(filename, population):
  os.makedirs("out", exist_ok=True)
  csv_path = os.path.join("out", filename)

  if not population:
    return

  # Assuming all solutions have the same number of objectives and variables
  pop_list = list(population)
  first_sol = pop_list[0]
  n_obj = first_sol.numberOfObjectives
  n_var = first_sol.numberOfDecisionVariables

  with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header: objectives followed by variables
    header = [f"obj_{i}" for i in range(n_obj)] + [f"var_{i}" for i in range(n_var)]
    writer.writerow(header)

    for sol in pop_list:
      row = list(sol.objectives) + list(sol.decisionVariables)
      writer.writerow(row)

  print(f"Results saved to {csv_path}")


def to_2d_array(values, width):
  array = np.asarray(values, dtype=float)
  if array.size == 0:
    return np.empty((0, width), dtype=float)
  if array.ndim == 1:
    return array.reshape(1, -1)
  return array


def save_dataset_csv(path, objectives, decision_vectors, n_obj, n_var):
  objectives = to_2d_array(objectives, n_obj)
  if decision_vectors is None:
    decision_vectors = np.full((len(objectives), n_var), np.nan, dtype=float)
  else:
    decision_vectors = to_2d_array(decision_vectors, n_var)

  fieldnames = [f"obj_{i}" for i in range(n_obj)]
  fieldnames.extend([f"var_{i}" for i in range(n_var)])

  with open(path, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(objectives)):
      row = dict()
      for j in range(n_obj):
        row[f"obj_{j}"] = float(objectives[i][j])
      for j in range(n_var):
        value = decision_vectors[i][j]
        row[f"var_{j}"] = "" if np.isnan(value) else float(value)
      writer.writerow(row)


def save_reference_points_csv(path, reference_points, n_obj):
  reference_points = to_2d_array(reference_points, n_obj)
  fieldnames = [f"obj_{i}" for i in range(n_obj)]

  with open(path, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for point in reference_points:
      row = {f"obj_{i}": float(point[i]) for i in range(n_obj)}
      writer.writerow(row)


def save_reference_front_csv(path, reference_front, n_obj):
  reference_front = to_2d_array(reference_front, n_obj)
  fieldnames = [f"obj_{i}" for i in range(n_obj)]

  with open(path, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=fieldnames)
    writer.writeheader()
    for point in reference_front:
      row = {f"obj_{i}": float(point[i]) for i in range(n_obj)}
      writer.writerow(row)


def save_dvl_results_csvs(run_dir, trace, reference_front, n_obj, n_var):
  csv_dir = run_dir / "csv"
  csv_dir.mkdir(parents=True, exist_ok=True)

  estimated_decision_vectors = trace.get("estimated_decision_vectors")
  if estimated_decision_vectors is None:
    estimated_decision_vectors = trace.get("estimated_decision_vectors_bounded")

  save_dataset_csv(csv_dir / "sample.csv", trace["sample_objectives"], trace["sample_decision_vectors"], n_obj, n_var)
  save_reference_points_csv(csv_dir / "reference_points.csv", trace["reference_points"], n_obj)
  save_dataset_csv(csv_dir / "estimated.csv", trace["estimated_objectives"], estimated_decision_vectors, n_obj, n_var)
  save_dataset_csv(csv_dir / "final_front.csv", trace["final_front_objectives"], trace["final_front_decision_vectors"], n_obj, n_var)
  save_reference_front_csv(csv_dir / "reference_front.csv", reference_front, n_obj)

  return csv_dir
