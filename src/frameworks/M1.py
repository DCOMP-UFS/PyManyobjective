import math
import numpy as np
from pyDOE import lhs
from smt.surrogate_models import KRG
import sys

from Problem import Problem
from Solution import Solution
from ParetoFront import ParetoFront

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

class M1():
    def __init__(self, problem, n_vars, p, tau, SEmax, EMO, mi, n_objs, EMO_gens):
        self.problem = problem  # objective function
        self.n_vars = n_vars  # number of variables
        self.p = p  # sample size
        self.tau = tau  # generations per metamodel
        self.SEmax = SEmax  # maximum high fidelity solution evaluations
        self.EMO = EMO  # multi-ojective evolutionary algorithm
        self.mi = mi  # EMO's population size

        self.n_objs = n_objs  # number of objectives
        self.EMO_gens = EMO_gens  # number of generations of EMO

    def run(self):
        ####################### FIXME ######################
        # check attributes
        # obj_function
        print(self.problem.numberOfObjectives)
        assert(self.problem.numberOfDecisionVariables == self.n_vars)
        assert(self.problem.numberOfObjectives == self.n_objs)
        # EMO
        assert(self.EMO.populationSize == self.mi)
        ####################################################

        t = 0
        k = (t % self.tau)
        Pt = []
        Pk = lhs_to_solution(lhs(self.n_vars, samples=self.p), self.n_objs, self.n_vars)  # initialize surrogate model's training set
        paretoFront = ParetoFront()

        eval = 0
        while (eval < self.SEmax):
            if t % self.tau == 0:
                Pk = list(set(Pk + Pt))
                print(Pk)

                # high-fidelity evaluations (functions)
                for p in Pk:
                    self.problem.evaluate(p)
                Fkm = []
                for i in range(self.n_objs):
                    Fkm.append([])
                    for p in Pk:
                        Fkm[i].append(p.objectives[i])
                X = []
                for p in Pk:
                    X.append(p.decisionVariables)

                eval = len(Pk)

                # Surrogate independently each objective function
                SMs = []  # surrogate models
                for i in range(self.n_objs):
                    SM = KRG(print_training=False, print_prediction=False)
                    SM.set_training_values(np.array(X), np.array(Fkm[i]))
                    SM.train()
                    SMs.append(SM)
                # update EMO to use the created surrogates as objective function
                # class SM_QF_layer above
                self.EMO.problem = SM_QF_layer(SMs, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

                if k == 0:
                    # Initialize EMO’s population
                    Pt = lhs_to_solution(lhs(self.n_vars, samples=self.mi), self.n_objs, self.n_vars)
                else:
                    paretoFront.fastNonDominatedSort(Pk)
                    Pt = []
                    i = j = 0
                    while len(Pt) < self.mi and j < len(paretoFront.getInstance().front):
                        Pt.append(paretoFront.getInstance().front[j][i])
                        i += 1
                        if i == len(paretoFront.getInstance().front[j]):
                            i = 0
                            j += 1
                k += 1

            # Optimize surrogate model
            self.EMO.execute()
            Pt = list(self.EMO.population)
            print(t + 1)
            t += 1

        Pfinal = list(set(Pk + Pt))

        # evaluate Pfinal using objective function
        for p in Pfinal:
            self.problem.evaluate(p)

        paretoFront.fastNonDominatedSort(Pfinal)
        return paretoFront.getInstance().front[0]