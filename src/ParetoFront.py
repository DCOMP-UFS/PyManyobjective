# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:32:09 2020

@author: jadso
"""

import numpy as np

class ParetoFront:
  def dominance(self, p1_obj, p2_obj):
    best_objectives = np.minimum(p1_obj, p2_obj)
    p1_optimal = np.all(p1_obj == best_objectives)
    p2_optimal = np.all(p2_obj == best_objectives)

    if p1_optimal and ~p2_optimal:
      return 1
    elif ~p1_optimal and p2_optimal:
      return -1
    return 0
    
  def fastNonDominatedSort(self, population):
    X = population.decisionVariables
    Y = np.copy(population.objectives)

    times_dominated = np.zeros(X.shape[0])
    dominated = np.full((X.shape[0], X.shape[0]), False)
    fronts = np.ones(X.shape[0])
    for i in range(X.shape[0]):
      possibly_better = np.any(Y < Y[i], axis=1)
      possibly_worse = np.any(Y > Y[i], axis=1)

      better = np.logical_and(possibly_better, ~possibly_worse)
      worse = np.logical_and(possibly_worse, ~possibly_better)

      times_dominated[i] += np.sum(better)
      dominated[i] = worse

    cur_not_dominated = times_dominated <= 0
    fronts[cur_not_dominated] = 0
    cur_front = 0
    visited = np.full(X.shape[0], False)
    while np.sum(cur_not_dominated) > 0:
      ac = np.sum(dominated[cur_not_dominated], axis=0)
      times_dominated -= ac
      visited = np.logical_or(visited, cur_not_dominated)
      cur_not_dominated = times_dominated <= 0
      cur_not_dominated = np.logical_and(cur_not_dominated, ~visited)
      fronts[cur_not_dominated] = cur_front + 1
      cur_front += 1
    
    return fronts
  
  def getBest(self, population):
    best_objectives = np.min(population.objectives, axis=0)
    sum_equals = np.sum(population.objectives == best_objectives, axis=1)
    return population.decisionVariables[np.argsort(sum_equals)[0]]