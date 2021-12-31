from src.problems.Problem import Problem
from src.Solution import Solution
import numpy as np
import math

class ZDT1(Problem):
    def __init__(self, numberOfDecisionVariables, decisionVariablesLimit=None):
        numberOfObjectives = 2

        lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
        decisionVariablesLimit = (lowerBounds, upperBounds)

        super(ZDT1, self).__init__(numberOfObjectives,
                                numberOfDecisionVariables,
                                decisionVariablesLimit)
        self.problem = "zdt1"

    def ideal_point(self):
        return np.array([0.0, 0.0])

    def evaluate(self, population):
        x = population.decisionVariables

        f1 = x[:, 0]
        g = 1 + 9.0 / (self.numberOfDecisionVariables - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        population.objectives = np.column_stack([f1, f2])

class ZDT3(ZDT1):
    def __init__(self, numberOfDecisionVariables):
        super().__init__(numberOfDecisionVariables)
        self.problem = "zdt3"

    def evaluate(self, population):
        x = population.decisionVariables

        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_vars - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        population.objectives = np.column_stack([f1, f2])

class ZDT6(ZDT1):
    def __init__(self, numberOfDecisionVariables):
        super().__init__(numberOfDecisionVariables)
        self.problem = "zdt6"
    
    def evaluate(self, population):
        x = population.decisionVariables

        f1 = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[:, 1:], axis=1) / (self.n_vars - 1.0), 0.25)
        f2 = g * (1 - np.power(f1 / g, 2))

        population.objectives = np.column_stack([f1, f2])
