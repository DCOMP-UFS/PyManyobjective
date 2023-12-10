import math
import numpy as np
from pyDOE2 import lhs
from sklearn.linear_model import LinearRegression
from smt.surrogate_models import KRG
import sys

from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront

from src.Population import Population, genPopulation

# manage Kriging input/output
class SM_QF_layer(Problem):
    def __init__(self, SMs, numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None):
        super(SM_QF_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit)
        self.SMs = SMs
    
    def evaluate(self, population):
        x = population.decisionVariables

        obj_solutions = self.SMs[0].predict_values(x)
        print(obj_solutions)
        print(self.SMs[1].predict_values(x))
        for i in range(1, len(self.SMs)):
            obj_solutions = np.hstack((obj_solutions, self.SMs[i].predict_values(x)))

        population.objectives = obj_solutions

class Scikit_layer(Problem):
    def __init__(self, SMs, numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit=None):
        super(Scikit_layer, self).__init__(numberOfObjectives, numberOfDecisionVariables, decisionVariablesLimit)
        self.SMs = SMs

    def evaluate(self, population):
        x = population.getNotEvaluatedVars()

        obj_solutions = np.transpose([self.SMs[0].predict(x)])
        for i in range(1, len(self.SMs)):
            print(self.SMs[i].predict(x))
            obj_solutions = np.c_[obj_solutions, self.SMs[i].predict(x)]

        population.setNotEvaluatedObjectives(obj_solutions)

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
        Pt = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        Pk = genPopulation(self.problem, samples=self.sample_size)
        paretoFront = ParetoFront()

        eval = 0
        while (eval < self.SEmax):
            if t % self.tau == 0:
                Pk.join(Pt)
                #check_limits(Pk, self.problem.decisionVariablesLimit)
                # High-fidelity evaluations (functions)
                self.problem.evaluate(Pk)
                Fkm = np.transpose(Pk.objectives)
                print(Fkm)

                if t == 0:
                    eval = self.sample_size
                else:
                    eval += self.EMO.populationSize

                # Surrogate independently each objective function
                SMs = []  # Surrogate models
                for i in range(self.problem.numberOfObjectives):
                    SM = LinearRegression(n_jobs=-1)
                    SM.fit(Pk.decisionVariables, Fkm[i])
                    SMs.append(SM)
                # Update EMO to use the created surrogates as objective function
                # Class SM_QF_layer above
                self.EMO.problem = Scikit_layer(SMs, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

                if k == 0:
                    # Initialize EMO’s population
                    Pt = genPopulation(self.problem, samples=self.EMO.populationSize, evaluate=False)
                else:
                    fronts = paretoFront.fastNonDominatedSort(Pk)
                    fronts_order = np.argsort(fronts)
                    X = np.copy(Pk.decisionVariables)
                    X = X[fronts_order][:self.EMO.populationSize]
                    Pt.decisionVariables = X
                k += 1

            # Optimize surrogate model
            self.EMO.evaluations = 0
            Pt.clearObjectives()
            self.EMO.execute(Pt)
            Pt = self.EMO.population
            Pt.clearObjectives()
            print("t:", t + 1)
            t += 1

        Pk.join(Pt)
        self.problem.evaluate(Pk)

        fronts = paretoFront.fastNonDominatedSort(Pk)
        Pk.filter(fronts == 0)
        return Pk
