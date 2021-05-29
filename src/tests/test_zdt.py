from ZDT1 import ZDT1

from pymoo.problems.multi.zdt import ZDT1 as ZDT1_pymoo
from pymoo.problems.multi.zdt import ZDT2 as ZDT2_pymoo
from pymoo.problems.multi.zdt import ZDT3 as ZDT3_pymoo
from pymoo.problems.multi.zdt import ZDT6 as ZDT6_pymoo

from pyDOE import lhs
from Solution import Solution

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

def run(n_vars, pop_size):
	zdt1 = ZDT1(n_vars)
	zdt1_pymoo = ZDT1_pymoo(n_vars)

	npP = lhs(n_vars, pop_size)
	npF = zdt1_pymoo.evaluate(npP)

	P = lhs_to_solution(npP, 2, n_vars)
	for p in P:
		zdt1.evaluate(p)

	for i in range(len(P)):
		print(P[i].objectives, npF[i])
