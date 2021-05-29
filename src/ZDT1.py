from Problem import Problem
from Solution import Solution
import numpy as np
import math

# Problemas da classe ZDT
class ZDT1(Problem):
  # Construtor
  def __init__(self, numberOfDecisionVariables, decisionVariablesLimit=None):
    numberOfObjectives = 2
    super(ZDT1,self).__init__(numberOfObjectives,
                               numberOfDecisionVariables,
                               decisionVariablesLimit)
    self.problem = "zdt1"
    
    lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
    upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]

  def evalG(self, solution):
    g = 0.0
    for i in range(1, solution.numberOfDecisionVariables):
      g += solution.decisionVariables[i]
    constant = 9.0 / (solution.numberOfDecisionVariables - 1)

    return constant * g + 1.0

  def evalH(self, f, g):
    h = 1.0 - math.sqrt(f / g)
    return h

  # Calcula os objetivos
  def evaluate(self, solution):
    f = [0] * solution.numberOfObjectives
    
    f[0] = solution.decisionVariables[0]
    g = self.evalG(solution)
    h = self.evalH(f[0], g)
    f[1] = h * g;

    solution.objectives[0] = f[0]
    solution.objectives[1] = f[1]

    return solution