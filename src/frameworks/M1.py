import math
import numpy as np
from pyDOE import lhs
from smt.surrogate_models import KRG
from fast_non_dominated_sort import fast_non_dominated_sort
import sys

# manage Kriging input/output
class SM_QF_layer():
    def __init__(self, SMs, n_vars, xl, xu):
        self.SMs = SMs
        self.n_vars = n_vars
        self.xl = xl
        self.xu = xu

    def evaluate(self, x):
        obj_solutions = self.SMs[0].predict_values(x)
        for i in range(1, len(self.SMs)):
            obj_solutions = np.hstack((obj_solutions, self.SMs[i].predict_values(x)))
        return obj_solutions

class M1():
    def __init__(self, f, n_vars, p, tau, SEmax, EMO, mi, n_objs, EMO_gens):
        self.f = f  # objective function
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
        assert (self.f.n_vars == self.n_vars)
        assert (self.f.n_objs == self.n_objs)
        # EMO
        assert (self.EMO.pop_size == self.mi)
        assert (self.EMO.n_objs == self.n_objs)
        ####################################################

        t = 0
        k = (t % self.tau)
        Pt = np.zeros((0, self.n_vars))
        Pk = lhs(self.n_vars, samples=self.p)  # initialize surrogate model's training set

        eval = 0
        while (eval < self.SEmax):
            if t % self.tau == 0:
                Pk = np.concatenate((Pk, Pt))
                Pk = np.unique(Pk, axis=0)

                # high-fidelity evaluations (functions)
                F_all = self.f.evaluate(Pk)
                Fkm = np.transpose(F_all)

                eval = len(Pk)

                # Surrogate independently each objective function
                SMs = []  # surrogate models
                for i in range(self.n_objs):
                    SM = KRG(print_training=False, print_prediction=False)
                    SM.set_training_values(Pk, Fkm[i])
                    SM.train()
                    SMs.append(SM)
                # update EMO to use the created surrogates as objective function
                # class SM_QF_layer above
                self.EMO.f = SM_QF_layer(SMs, self.n_vars, self.f.xl, self.f.xu)

                if k == 0:
                    # Initialize EMO’s population
                    Pt = lhs(self.n_vars, samples=self.mi)
                else:
                    ranks = fast_non_dominated_sort(len(Pk),
                                                    F_all)  # any implementation of non-dominated sort using python list of lists will work
                    Q = []
                    i = j = 0
                    while len(Q) < self.mi and j < len(ranks):
                        Q.append(ranks[j][i])
                        i += 1
                        if i == len(ranks[j]):
                            i = 0
                            j += 1
                    Pt = Pk[Q]
                k += 1

            # Optimize surrogate model
            Pt = self.EMO.run(Pt, n_gens=self.EMO_gens)
            print(t + 1)
            t += 1

        Pfinal = np.concatenate((Pk, Pt))

        # evaluate Pfinal using objective function
        F = self.f.evaluate(Pfinal)

        inds = fast_non_dominated_sort(len(Pfinal), F)[0]  # take indexes of pareto front

        return Pfinal[inds]