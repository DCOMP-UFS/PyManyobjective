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
        super(SM_QF_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit)
        self.SMs = SMs

    def evaluate(self, solution):
        objectives = []
        for i in range(len(self.SMs)):
            objectives.append(self.SMs[i].predict_values(np.array([solution.decisionVariables]))[0][0])
        solution.objectives = objectives
        return solution

def lhs_to_solution(A, limits, numberOfObjectives, numberOfDecisionVariables):
    for i in range(numberOfDecisionVariables):
        A[:, i] = A[:, i] * (limits[1][i] - limits[0][i]) + limits[0][i]

    B = list()
    for a in A:
        b = Solution(numberOfObjectives, numberOfDecisionVariables)
        decisionVariables = []
        for x in a:
            decisionVariables.append(x)
        b.decisionVariables = decisionVariables
        B.append(b)
    return B

def check_limits(solutions, limits):
    print("-----------------------------------")
    print("checking...")
    print("-----------------------------------")
    j = 0
    for solution in solutions:
        for i in range(len(limits[0])):
            if (solution.decisionVariables[i] < limits[0][i] or solution.decisionVariables[i] > limits[1][i]):
                print(j, solution.decisionVariables[i])
                print(limits[0][i], limits[1][i])
                print("OOOPPSS!!")
                raise Exception
        j += 1

class M1():
    def __init__(self, problem, EMO, sample_size, tau, SEmax):
        self.problem = problem # Objective function
        self.EMO = EMO # Multi-ojective evolutionary algorithm
        self.sample_size = sample_size # Random initial population size
        self.tau = tau # Generations per metamodel
        self.SEmax = SEmax # Maximum high fidelity solution evaluations

    def run(self):
        t = 0
        k = (t % self.tau)
        Pt = []
        Pk = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.sample_size), self.problem.decisionVariablesLimit, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)  # Initialize surrogate model's training set
        paretoFront = ParetoFront()

        eval = 0
        while (eval < self.SEmax):
            if t % self.tau == 0:
                Pk = list(set(Pk + Pt))
                check_limits(Pk, self.problem.decisionVariablesLimit)
                # High-fidelity evaluations (functions)
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

                eval = len(Pk)

                # Surrogate independently each objective function
                SMs = []  # Surrogate models
                for i in range(self.problem.numberOfObjectives):
                    SM = KRG(print_training=False, print_prediction=False)
                    SM.set_training_values(np.array(X), np.array(Fkm[i]))
                    SM.train()
                    SMs.append(SM)
                # Update EMO to use the created surrogates as objective function
                # Class SM_QF_layer above
                self.EMO.problem = SM_QF_layer(SMs, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

                if k == 0:
                    # Initialize EMO’s population
                    Pt = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.EMO.populationSize), self.problem.decisionVariablesLimit, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
                else:
                    paretoFront.fastNonDominatedSort(Pk)
                    Pt = []
                    i = j = 0
                    while len(Pt) < self.EMO.populationSize and j < len(paretoFront.getInstance().front):
                        Pt.append(paretoFront.getInstance().front[j][i])
                        i += 1
                        if i == len(paretoFront.getInstance().front[j]):
                            i = 0
                            j += 1
                k += 1

            # Optimize surrogate model
            self.EMO.evaluations = 0
            self.EMO.execute(set(Pt))
            Pt = list(self.EMO.population)
            print(t + 1)
            t += 1

        Pfinal = list(set(Pk + Pt))

        # Evaluate Pfinal using objective function
        for p in Pfinal:
            self.problem.evaluate(p)

        paretoFront.fastNonDominatedSort(Pfinal)
        return paretoFront.getInstance().front[0]