from src.problems.Problem import Problem
from src.Solution import Solution
import numpy as np
import math

class ZDT1(Problem):
    def __init__(self, numberOfDecisionVariables, decisionVariablesLimit=None):
        numberOfObjectives = 2
        super(ZDT1, self).__init__(numberOfObjectives,
                                numberOfDecisionVariables,
                                decisionVariablesLimit)
        self.problem = "zdt1"
        
        lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]

        self.decisionVariablesLimit = (lowerBounds, upperBounds)

    def ideal_point(self):
        return np.array([0.0, 0.0])

    def evalG(self, solution):
        g = 0.0
        for i in range(1, solution.numberOfDecisionVariables):
            g += solution.decisionVariables[i]
        constant = 9.0 / (solution.numberOfDecisionVariables - 1)

        return constant * g + 1.0

    def evalH(self, f, g):
        h = 1.0 - math.sqrt(f / g)
        return h

    def evaluate(self, solution):
        f = [0] * solution.numberOfObjectives
    
        f[0] = solution.decisionVariables[0]
        g = self.evalG(solution)
        h = self.evalH(f[0], g)
        f[1] = h * g

        solution.objectives[0] = f[0]
        solution.objectives[1] = f[1]

        return solution

class ZDT3(ZDT1):
    def __init__(self, numberOfDecisionVariables):
        super().__init__(numberOfDecisionVariables)
        self.problem = "zdt3"

    def evalH(self, f, g):
        h = 1.0 - math.sqrt(f / g) - (f / g) * math.sin(10 * math.pi * f)
        return h

class ZDT6(ZDT1):
    def __init__(self, numberOfDecisionVariables):
        super().__init__(numberOfDecisionVariables)
        self.problem = "zdt6"
    
    def evalG(self, solution):
        g = 0.0
        for i in range(1, solution.numberOfDecisionVariables):
            g += solution.decisionVariables[i]
        g = g / (solution.numberOfDecisionVariables - 1)
        g = math.pow(g, 0.25)
        g = 9.0 * g
        g = 1.0 + g
        return g

    def evalH(self, f, g):
        return 1.0 - math.pow((f / g), 2.0)

    def evaluate(self, solution):
        f = [0] * solution.numberOfObjectives
    
        x1 = solution.decisionVariables[0]
        f[0] = 1 - math.exp(-4 * x1) * math.pow(math.sin(6 * math.pi * x1), 6)
        g = self.evalG(solution)
        h = self.evalH(f[0], g)
        f[1] = h * g

        solution.objectives[0] = f[0]
        solution.objectives[1] = f[1]

        return solution
