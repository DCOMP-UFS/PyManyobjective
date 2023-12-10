import math
import numpy as np
from pyDOE2 import lhs
from smt.surrogate_models import KRG
import sys

from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront

from src.Population import Population
from src.Population import genPopulation

# manage Kriging input/output
class SM_QF_layer(Problem):
    def __init__(self, SMs, numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None):
        super(SM_QF_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit)
        self.SMs = SMs

    def evaluate(self, solution):
        objectives = []
        for i in range(len(self.SMs)):
            objectives.append(self.SMs[i].predict_values(np.array([solution.decisionVariables]))[0][0])
        solution.objectives = objectives
        return solution
    
    def evaluate(self, population):
        x = population.decisionVariables

        obj_solutions = self.SMs[0].predict_values(x)
        for i in range(1, len(self.SMs)):
            obj_solutions = np.hstack((obj_solutions, self.SMs[i].predict_values(x)))

        population.objectives = obj_solutions

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
        for i in range(P.objectives.shape[0]):
            f = P.objectives[i]
            f_norm = f / np.linalg.norm(f)
            dist = np.linalg.norm(np.cross(np.array(r), -f_norm)) / np.linalg.norm(np.array(r))
            dists.append(dist)
        return dists

    def ASF(self, P, dir_index):
        return np.max(P.objectives - self.R[dir_index], axis=1)

    def run(self):
        P = genPopulation(self.problem, self.sample_size)
        paretoFront = ParetoFront()

        eval = self.sample_size
        while eval < self.SEmax:
            for dir_index in range(len(self.R)):
                self.problem.evaluate(P)

                # sort P according to distance from r and select nearest solutions
                dists = self.dist_func(P, np.array(self.R[dir_index]))
                dists_order = np.argsort(dists)

                Pr = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
                Pr.decisionVariables = P.decisionVariables[dists_order]
                Pr.objectives = P.objectives[dists_order]
                Pr.decisionVariables = Pr.decisionVariables[:max(1, int(self.alfa * P.decisionVariables.shape[0]))]
                Pr.objectives = Pr.objectives[:max(1, int(self.alfa * P.decisionVariables.shape[0]))]

                ASFs = self.ASF(Pr, dir_index)

                SM = KRG(print_training=False, print_prediction=False)
                SM.set_training_values(Pr.decisionVariables, ASFs)
                SM.train()
                SMs = [SM]
                # 1 objective
                self.EMO.problem = SM_QF_layer(SMs, 1, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

                Pr.numberOfObjectives = 1
                Pr.objectives = np.full((Pr.decisionVariables.shape[0], 1), np.array([0]))

                for _ in range(0, self.k):
                    self.EMO.evaluations = 0
                    self.EMO.execute(Pr)
                    for x in self.EMO.population.decisionVariables:
                        is_in = np.any(np.all(x == P.decisionVariables, axis=1))
                        if not is_in:
                            P.decisionVariables = np.append(P.decisionVariables, np.array([x]), axis=0)
                            break

                    eval += 1
                    print(P.decisionVariables.shape[0])
                    if eval >= self.SEmax:
                        self.problem.evaluate(P)
                        fronts = paretoFront.fastNonDominatedSort(P)
                        P.decisionVariables = P.decisionVariables[fronts == 0]
                        P.objectives = P.objectives[fronts == 0]
                        return P

        P.numberOfObjectives = self.problem.numberOfObjectives
        self.problem.evaluate(P)
        fronts = paretoFront.fastNonDominatedSort(P)
        P.decisionVariables = P.decisionVariables[fronts == 0]
        P.objectives = P.objectives[fronts == 0]
        return P
