from src.problems.DTLZ import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7

from pymoo.problems.many.dtlz import DTLZ1 as DTLZ1_pymoo
from pymoo.problems.many.dtlz import DTLZ2 as DTLZ2_pymoo
from pymoo.problems.many.dtlz import DTLZ3 as DTLZ3_pymoo
from pymoo.problems.many.dtlz import DTLZ4 as DTLZ4_pymoo
from pymoo.problems.many.dtlz import DTLZ5 as DTLZ5_pymoo
from pymoo.problems.many.dtlz import DTLZ6 as DTLZ6_pymoo
from pymoo.problems.many.dtlz import DTLZ7 as DTLZ7_pymoo

from pyDOE import lhs
from src.Solution import Solution

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

def run(dummy, pop_size):
    n_vars = 12
    dtlz = DTLZ2()
    dtlz_pymoo = DTLZ2_pymoo(12, 3)

    npP = lhs(n_vars, pop_size)
    npF = dtlz_pymoo.evaluate(npP)

    P = lhs_to_solution(npP, 3, n_vars)
    for p in P:
        dtlz.evaluate(p)

    success = True
    for i in range(len(P)):
        print(P[i].objectives, npF[i])
        for j in range(dtlz.numberOfObjectives):
            dif = abs(P[i].objectives[j] - npF[i][j])
            if dif > 0.00001:
                success = False

    if success:
        print("equals")
    else:
        print("not equals")
