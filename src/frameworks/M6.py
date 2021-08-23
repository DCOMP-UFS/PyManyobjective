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

class M6():
    def __init__(self, problem, problem_np, EMO, sample_size, SEmax, R, kktpm):
        self.problem = problem # objective function
        self.problem_np = problem_np
        self.sample_size = sample_size # sample size
        self.SEmax = SEmax # maximum high fidelity solution evaluations
        self.R = R # reference direction set
        self.EMO = EMO
        self.kktpm = kktpm

    def clustering(self, population):
        for i in range(len(population)):
            best_dist = None
            cluster_id = -1
            objs = np.array(population[i].objectives)
            for j in range(len(self.R)):
                r = np.array(self.R[j])
                dist = np.linalg.norm(np.cross(r, -objs)) / np.linalg.norm(r)
                if cluster_id == -1:
                    best_dist = dist
                    cluster_id = j
                elif dist < best_dist:
                    best_dist = dist
                    cluster_id = j
            population[i].cluster = cluster_id

    def run(self):
        Pk = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.sample_size), self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        paretoFront = ParetoFront()

        eval = 0
        while (eval < self.SEmax):
            for p in Pk:
                self.problem.evaluate(p)
            self.clustering(Pk)

            X = []
            for p in Pk:
                X.append(p.decisionVariables)

            fitness = self.kktpm.calc(X=np.array(X), problem=self.problem_np, ideal_point=self.problem.ideal_point())

            eval = len(Pk)

            not_nan = ~np.isnan(fitness.flatten())

            SM = KRG(print_training=False, print_prediction=False)
            SM.set_training_values(np.array(X)[not_nan], fitness[not_nan])
            SM.train()
            SMs = [SM]
            self.EMO.problem = SM_QF_layer(SMs, 1, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

            self.EMO.evaluations = 0
            self.EMO.execute(set(Pk))
            Pt = []
            for p in self.EMO.population:
                s = Solution(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
                s.decisionVariables = p.decisionVariables
                Pt.append(s)


            if len(Pt) + eval > self.SEmax:
                Pt = Pt[:self.SEmax - eval]
            Pk = list(set(Pk + Pt))
            eval = len(Pk)

        Pfinal = Pk

        # evaluate Pfinal using objective function
        for p in Pfinal:
            self.problem.evaluate(p)

        paretoFront.fastNonDominatedSort(Pfinal)
        return paretoFront.getInstance().front[0]