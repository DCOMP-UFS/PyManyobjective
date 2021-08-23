import math
import numpy as np
from pyDOE import lhs
from smt.surrogate_models import KRG
import sys

from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront

# manage Kriging input/output
class SM_QF_layer(Problem):
    def __init__(self, SMs, numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None):
        super(SM_QF_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None)
        self.SMs = SMs

        lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
    
        self.decisionVariablesLimit = (lowerBounds, upperBounds)

    def evaluate(self, solution):
        objectives = []
        for i in range(len(self.SMs)):
            objectives.append(self.SMs[i].predict_values(np.array([solution.decisionVariables]))[0][0])
        solution.objectives = objectives
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

class M3():
    def __init__(self, problem, EMO, sample_size, SEmax, k, alfa, R):
        self.problem = problem # Objective function
        self.EMO = EMO
        self.sample_size = sample_size # Sample size
        self.SEmax = SEmax # Maximum high fidelity solution evaluations
        self.k = k # Number of points created for each reference direciton
        self.alfa = alfa # Fraction of samples used for metamodel
        self.R = R # Reference direction set

    def dist_func(self, P, r):
        dists = []
        for p in P:
            f = np.array(p.objectives)
            f_norm = f / np.linalg.norm(f)
            dist = np.linalg.norm(np.cross(np.array(r), -f_norm)) / np.linalg.norm(np.array(r))
            dists.append(dist)
        return dists

    def ASF(self, P, dir_index):
        F = []
        for p in P:
            F.append(p.objectives)
        F = np.array(F)

        return np.max(F - self.R[dir_index], axis=1)

    def run(self):
        P = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.sample_size), self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        paretoFront = ParetoFront()

        eval = self.sample_size
        while eval < self.SEmax:
            for dir_index in range(len(self.R)):
                for p in P:
                    self.problem.evaluate(p)

                # sort P according to distance from r and select nearest solutions
                dists = self.dist_func(P, np.array(self.R[dir_index]))
                dists_order = np.argsort(dists)
                Pr = [P[i] for i in dists_order][:max(1, int(self.alfa * len(P)))]

                ASFs = self.ASF(Pr, dir_index)

                # Pr with ASF as objective
                for i in range(len(Pr)):
                    pr = Solution(1, self.problem.numberOfDecisionVariables)
                    pr.decisionVariables = Pr[i].decisionVariables
                    Pr[i] = pr

                Xr = []
                for pr in Pr:
                    Xr.append(pr.decisionVariables)

                SM = KRG(print_training=False, print_prediction=False)
                SM.set_training_values(np.array(Xr), np.array(ASFs))
                SM.train()
                SMs = [SM]
                # 1 objective
                self.EMO.problem = SM_QF_layer(SMs, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

                for i in range(0, self.k):
                    self.EMO.evaluations = 0
                    self.EMO.execute(set(Pr)) # best found solution
                    for sol in list(self.EMO.population):
                        if sol not in P:
                            non_ASF_sol = Solution(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
                            non_ASF_sol.decisionVariables = sol.decisionVariables
                            P.append(non_ASF_sol)
                            break

                    eval += 1
                    print(len(P))
                    if eval >= self.SEmax:
                        for p in P:
                            self.problem.evaluate(p)
                        paretoFront.fastNonDominatedSort(P)
                        return paretoFront.getInstance().front[0]

        #evaluate Pfinal using objective function
        for p in P:
            self.problem.evaluate(p)
        paretoFront.fastNonDominatedSort(P)
        return paretoFront.getInstance().front[0]
