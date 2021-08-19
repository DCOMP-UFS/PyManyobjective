from tests.M1_test import run
from ZDT import ZDT1

problem = ZDT1(numberOfDecisionVariables=10)

run(10, problem, "tests/zdt1_front.txt", print_igd=True, print_y=False, print_x=False, show_graph=True)