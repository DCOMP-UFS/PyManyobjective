import math
import random
import numpy as np
from pyDOE import lhs
from pyearth import Earth

from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront

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

    def generate_candidate_points(self, probability, bestSolution):
        z = []

        for k in range(self.m):
            r = []
            index = []
            for i in range(self.d):
                r.append(random.random())
                if (r[i] < probability):
                    index.append(i)
            if len(index) == 0:
                index.append(random.randrange(self.d))

            zk = bestSolution.clone()
            for i in range(len(index)):
                zk.decisionVariables[i] += random.gauss(0, self.variance * self.variance)
                # repair
                if zk.decisionVariables[i] < self.problem.decisionVariablesLimit[0][i]:
                    zk.decisionVariables[i] = self.problem.decisionVariablesLimit[0][i]
                elif zk.decisionVariables[i] > self.problem.decisionVariablesLimit[1][i]:
                    zk.decisionVariables[i] = self.problem.decisionVariablesLimit[1][i]
            z.append(zk)

        return z

    def select_evaluation_point(self, candidates, SM, D):
        Rn_min = None
        Rn_max = None

        X = []
        for p in candidates:
            X.append(p.decisionVariables)
        Rn_z = SM.predict(np.array(X))
        for i in range(len(candidates)):
            # compute Rn(z)min
            if Rn_min == None or Rn_z[i] < Rn_min:
                Rn_min = Rn_z[i]
            # compute Rn(z)max
            if Rn_max == None or Rn_z[i] > Rn_max:
                Rn_max = Rn_z[i]

        V = []
        for i in range(len(candidates)):
            if Rn_min == Rn_max:
                V.append(1)
            else:
                v = (Rn_z[i] - Rn_min) / (Rn_max - Rn_z[i])
                V.append(v)

        dis = []
        dis_min = None
        dis_max = None
        for z in candidates:
            z_dis = None
            for i in range(len(D)):
                calculated_dis = z.calc_dist(D[i])
                if z_dis == None or calculated_dis < z_dis:
                    z_dis = calculated_dis

            if dis_min == None or z_dis < dis_min:
                dis_min = z_dis
            if dis_max == None or z_dis > dis_max:
                dis_max = z_dis

            dis.append(z_dis)

        W = []
        for i in range(len(candidates)):
            if dis_min == dis_max:
                W.append(1)
            else:
                w = (dis_max - dis[i]) / (dis_max - dis_min)
                W.append(w)

        S = []
        for i in range(len(candidates)):
            s = self.wv * V[i] + self.ww * W[i]
            S.append(s)

        s_min = None
        best_z = None
        for i in range(len(candidates)):
            if s_min == None or S[i] < s_min:
                s_min = S[i]
                best_z = candidates[i]

        return best_z

    def update_variance(self):
        if self.T1 > self.thr1:
            self.variance *= 2
            self.T1 = 0

        if self.T2 > self.thr2:
            self.variance = max(self.variance / 2, 0) ### FIXME ###
            self.T2 = 0

    def run(self):
        self.d = self.problem.numberOfDecisionVariables
        self.m = 100 * self.d
        self.n = 2 * self.d + 1
        self.prob = min(20 / self.d, 1)
        assert(self.problem.numberOfObjectives == 1)

        Pk = lhs_to_solution(lhs(self.problem.numberOfDecisionVariables, samples=self.n), self.problem.decisionVariablesLimit, self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)  # Initialize surrogate model's training set
        paretoFront = ParetoFront()

        for p in Pk:
            self.problem.evaluate(p)

        for iterator in range(self.Gmax):
            Fkm = []
            for i in range(self.problem.numberOfObjectives):
                Fkm.append([])
                for p in Pk:
                    Fkm[i].append(p.objectives[i])
            X = []
            for p in Pk:
                X.append(p.decisionVariables)

            SMs = []
            for i in range(self.problem.numberOfObjectives):
                SM = Earth()
                SM.fit(np.array(X), np.array(Fkm[i]))
                SMs.append(SM)

            bestSolution = Pk[0]
            for i in range(1, self.n):
                if self.paretoFront.dominance(Pk[i], bestSolution) == self.paretoFront.W:
                    bestSolution = Pk[i]

            cur_prob = self.prob * (1 - math.log(iterator + 1) / math.log(self.Gmax))

            candidate_points = self.generate_candidate_points(cur_prob, bestSolution)
            promising_point = self.select_evaluation_point(candidate_points, SMs[0], Pk)
            self.problem.evaluate(promising_point)
            Pk.append(promising_point)
            self.n += 1
            self.update_variance()

            #print(iterator)

        paretoFront.fastNonDominatedSort(Pk)
        return paretoFront.getInstance().front[0]