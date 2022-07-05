import math
import random
import numpy as np
import scipy
from pyDOE2 import lhs
from pyearth import Earth

from src.problems.Problem import Problem
from src.ParetoFront import ParetoFront
from src.Population import Population
from src.Population import genPopulation

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

class MARSAOP():
    def __init__(self, problem, Gmax, prob, variance, wv, thr1, thr2):
        self.problem = problem # Objective function
        self.Gmax = Gmax
        #self.prob = prob
        self.variance = variance

        self.T1 = 0
        self.thr1 = thr1
        self.T2 = 0
        self.thr2 = thr2

        self.wv = wv
        self.ww = 1 - wv

        self.paretoFront = ParetoFront()

        assert(self.wv >= 0 and self.wv <= 1)
        assert(self.ww >= 0 and self.ww <= 1)
        assert(self.wv + self.ww == 1)

    def generate_candidate_points(self, probability, bestVariables):
        z = []

        random_probs = np.random.random((self.m, self.d))
        do_change = random_probs < probability

        without_change = np.sum(do_change, axis=1) == 0
        # FIXME: is this iteration necessary?
        without_change_ids = np.arange(stop=self.m)[without_change]
        selected_variables = np.random.randint(0, self.d, self.m)
        for id in without_change_ids:
            do_change[id][selected_variables[id]] = True

        change = np.random.normal(0, self.variance * self.variance, (self.m, self.d))

        X = np.tile(bestVariables, (self.m, 1))
        X[do_change] += change[do_change]
        # repair X
        np.clip(X, self.problem.decisionVariablesLimit[0], self.problem.decisionVariablesLimit[1], X)

        candidate_points = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        candidate_points.decisionVariables = X

        return candidate_points

    def select_evaluation_point(self, candidates, SM, D):
        Rn_min = None
        Rn_max = None

        X = candidates.decisionVariables
        Rn_z = SM.predict(X)
        Rn_min = np.min(Rn_z)
        Rn_max = np.max(Rn_z)

        V = np.zeros(X.shape[0])
        can_div = Rn_min != Rn_max
        V[~can_div] = 1
        V[can_div] = (Rn_z[can_div] - Rn_min) / (Rn_max - Rn_min)

        all_dis = scipy.spatial.distance.cdist(X, D.decisionVariables)
        dis = np.min(all_dis, axis=1)
        dis_min = np.min(dis)
        dis_max = np.max(dis)

        W = np.zeros(candidates.decisionVariables.shape[0])
        can_div = dis_min != dis_max
        W[~can_div] = 1
        W[can_div] = (-dis[can_div] + dis_max) / (dis_max - dis_min)

        S = self.wv * V + self.ww * W
        return X[np.argmin(S)]

    def update_variance(self):
        if self.T1 > self.thr1:
            self.variance *= 2
            self.T1 = 0

        if self.T2 > self.thr2:
            self.variance = max(self.variance / 2, 0) ### FIXME ###
            self.T2 = 0

    def run(self):
        self.d = self.problem.numberOfDecisionVariables
        self.m = 3
        self.n = 2 * self.d + 1
        self.prob = min(20 / self.d, 1)
        assert(self.problem.numberOfObjectives == 1)

        bestPrecision = None
        Pk = genPopulation(self.problem, self.n)
        paretoFront = ParetoFront()

        self.problem.evaluate(Pk)

        for iterator in range(self.Gmax):
            print(iterator)

            Fkm = np.transpose(Pk.objectives)

            X = Pk.decisionVariables

            SMs = []
            for i in range(self.problem.numberOfObjectives):
                SM = Earth()
                SM.fit(X, Fkm[i])
                SMs.append(SM)

            bestVariables = paretoFront.getBest(Pk)

            cur_prob = self.prob * (1 - math.log(iterator + 1) / math.log(self.Gmax))

            candidate_points = self.generate_candidate_points(cur_prob, bestVariables)
            promising_point = self.select_evaluation_point(candidate_points, SMs[0], Pk)
            Pk.add(promising_point)
            self.problem.evaluate(Pk)

            if bestPrecision == None or Pk.objectives[-1][0] < bestPrecision:
                bestPrecision = Pk.objectives[-1][0]
                self.T1 += 1
                self.T2 = 0
            else:
                self.T2 += 1
                self.T1 = 0

            self.n += 1
            self.update_variance()

            #print(iterator)

        fronts = paretoFront.fastNonDominatedSort(Pk)
        Pk.filter(fronts == 0)
        return Pk