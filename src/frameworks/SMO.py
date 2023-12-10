import math
import numpy as np
from pyDOE2 import lhs
from sklearn.ensemble import RandomForestRegressor
import sys

from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront

class Scikit_layer(Problem):
    def __init__(self, SMs, numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None):
        super(Scikit_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None)
        self.SMs = SMs

        lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
    
        self.decisionVariablesLimit = (lowerBounds, upperBounds)

    def evaluate(self, solution):
        if not solution.evaluated:
            objectives = []
            for i in range(len(self.SMs)):
                objectives.append(self.SMs[i].predict(np.array([solution.decisionVariables]))[0])
            solution.objectives = objectives
            solution.evaluated = True
        return solution

def lhs_to_solution(A, numberOfObjectives, numberOfDecisionVariables):
    B = list()
    for a in A:
        b = Solution(numberOfObjectives, numberOfDecisionVariables)
        decisionVariables = []
        for x in a:
            decisionVariables.append(x)
        b.decisionVariables = decisionVariables
        B.append(b)
    return B

class SMO():
    def __init__(self, problem, sample_size, t_max, t_parada, EMO):
        self.problem = problem  # objective function
        self.sample_size = sample_size  # sample size
        self.t_max = t_max  # maximum high fidelity solution evaluations
        self.t_parada = t_parada
        self.EMO = EMO  # multi-ojective evolutionary algorithm

    def run(self):
        Pt = []
        Pk = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.sample_size), self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)  # initialize surrogate model's training set
        paretoFront = ParetoFront()

        SMs = [] # surrogate models
        for _ in range(self.problem.numberOfObjectives):
            SMs.append(RandomForestRegressor(n_estimators=0, max_depth=None, min_samples_split=2, random_state=0, criterion='friedman_mse', n_jobs=1, warm_start=True))

        eval = self.sample_size
        while (eval <= self.t_parada):
            Pk = list(set(Pk + Pt))

            if eval <= self.t_max:
                # high-fidelity evaluations (functions)
                for p in Pk:
                    self.problem.evaluate(p)
                Fkm = []
                for i in range(self.problem.numberOfObjectives):
                    Fkm.append([])
                    for p in Pk:
                        Fkm[i].append(p.objectives[i])
                X = []
                for p in Pk:
                    X.append(p.decisionVariables)

                for i in range(self.problem.numberOfObjectives):
                    SMs[i].n_estimators += 35
                    SMs[i].fit(X, Fkm[i])
            else:
                self.EMO.problem = Scikit_layer(SMs, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

            Pt = []
            paretoFront.fastNonDominatedSort(Pk)
            for front in paretoFront.getInstance().front:
                if len(Pt) + len(front) > self.EMO.populationSize:
                    Pt += front[:self.EMO.populationSize - len(Pt)]
                    break
                Pt += front

            self.EMO.evaluations = 0
            self.EMO.execute(set(Pk))
            Pk = list(self.EMO.population)
            eval += self.EMO.populationSize
            print(eval)

        Pfinal = list(set(Pk + Pt))

        # evaluate Pfinal using objective function
        for p in Pfinal:
            self.problem.evaluate(p)

        paretoFront.fastNonDominatedSort(Pfinal)
        return paretoFront.getInstance().front[0]