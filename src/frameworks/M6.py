import math
import numpy as np
from pyDOE2 import lhs
from smt.surrogate_models import KRG
import sys
from src.Population import genPopulation

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
    
    def evaluate(self, population):
        x = population.decisionVariables

        obj_solutions = self.SMs[0].predict_values(x)
        for i in range(1, len(self.SMs)):
            obj_solutions = np.hstack((obj_solutions, self.SMs[i].predict_values(x)))

        population.objectives = obj_solutions

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
        P_obj = population.objectives

        P_cluster = []
        for i in range(len(P_obj)):
            best_dist = None
            cluster_id = -1
            for j in range(len(self.R)):
                dist = np.linalg.norm(np.cross(np.array(self.R[j]), -P_obj[i])) / np.linalg.norm(np.array(self.R[j]))
                if cluster_id == -1:
                    best_dist = dist
                    cluster_id = j
                elif dist < best_dist:
                    best_dist = dist
                    cluster_id = j
            P_cluster.append(cluster_id)
        
        population.cluster = P_cluster

    def run(self):
        Pk = genPopulation(self.problem, self.sample_size)
        paretoFront = ParetoFront()

        eval = self.sample_size
        while (eval < self.SEmax):
            self.problem.evaluate(Pk)
            self.clustering(Pk)

            X = Pk.decisionVariables

            fitness = self.kktpm.calc(X=X, problem=self.problem_np, ideal_point=self.problem.ideal_point())

            not_nan = ~np.isnan(fitness.flatten())

            SM = KRG(print_training=False, print_prediction=False)
            SM.set_training_values(X[not_nan], fitness[not_nan])
            SM.train()
            SMs = [SM]
            self.EMO.problem = SM_QF_layer(SMs, 1, self.problem.numberOfDecisionVariables, self.problem.decisionVariablesLimit)

            Pk.numberOfObjectives = 1
            Pk.objectives = np.full((Pk.decisionVariables.shape[0], 1), np.array([0]))

            self.EMO.evaluations = 0
            self.EMO.execute(Pk)
            Pt = self.EMO.population

            Pk.numberOfObjectives = self.problem.numberOfObjectives

            if Pt.decisionVariables.shape[0] + eval > self.SEmax:
                Pt.decisionVariables = Pt.decisionVariables[:self.SEmax - eval]
            Pk.join(Pt)
            eval += Pt.decisionVariables.shape[0]

        self.problem.evaluate(Pk)

        fronts = paretoFront.fastNonDominatedSort(Pk)
        Pk.filter(fronts == 0)
        return Pk