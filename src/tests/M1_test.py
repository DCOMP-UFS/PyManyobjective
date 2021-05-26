from M1 import M1

import numpy as np

from algorithms.NSGA2 import NSGA2

import matplotlib.pyplot as plt
from pymoo.factory import get_performance_indicator

def run(n, f, comp_file, print_igd, print_x, print_y, show_graph):
    assert(n == f.n_vars)

    nsga2 = NSGA2(pop_size=100, n_objs=2, f=f)

    m1 = M1(f=f, n_vars=10, p=300, tau=50, SEmax=500, EMO=nsga2, mi=100, n_objs=2, EMO_gens=100)
    my_front = m1.run()

    true_front = []
    with open(comp_file, 'r') as zdt_file:
        for line in zdt_file.readlines():
            xy = [float(line.split()[0]), float(line.split()[1])]
            true_front.append(xy)

    F = f.evaluate(my_front)

    if print_igd:
        igd_p = get_performance_indicator("igd", np.array(true_front))
        print(igd_p.calc(F))
        print("")

    if print_x:
        for x in my_front:
            print(x)
        print("")

    if print_y:
        for y in F:
            print(y)
        print("")

    F_x = []
    F_y = []
    for xy in F:
        F_x.append(xy[0])
        F_y.append(xy[1])

    true_x = []
    true_y = []
    for xy in true_front:
        true_x.append(xy[0])
        true_y.append(xy[1])

    if show_graph:
        plt.plot(F_x, F_y, 'bo')
        plt.plot(true_x, true_y, 'ro', markersize=2)
        plt.axis([0, 1.0, 0, 1.2])
        plt.show()

    return igd_p.calc(F)
