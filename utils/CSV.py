import csv
import os


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