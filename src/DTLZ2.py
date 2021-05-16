#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:16:34 2021

@author: jad
"""

from Problem import Problem
import numpy as np

# Problemas da classe DTLZ
class DTLZ2(Problem):
  # Construtor
  def __init__(self, numberOfObjectives=3, k=5, decisionVariablesLimit=None):
    numberOfDecisionVariables = k + numberOfObjectives - 1
    super(DTLZ2,self).__init__(numberOfObjectives,
                               numberOfDecisionVariables,
                               decisionVariablesLimit)
    self.problem = "dtlz1"
    
    lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
    upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
    
    self.decisionVariablesLimit = (lowerBounds, upperBounds)
  
  # Calcula os objetivos
  def evaluate(self, solution):
    numberOfDecisionVariables = self.numberOfDecisionVariables
    numberOfObjectives        = solution.numberOfObjectives;
    
    f = [0.0 for _ in range(numberOfObjectives)];
    x = [0.0 for _ in range(numberOfDecisionVariables)] ;

    for i in range(numberOfDecisionVariables):
      x[i] = solution.decisionVariables[i]

    k = numberOfDecisionVariables - numberOfObjectives + 1;

    g = 0.0
    for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
      g += (x[i] - 0.5) * (x[i] - 0.5)

    for i in range(numberOfObjectives):
      f[i] = 1.0 + g

    for i in range(numberOfObjectives):
      for j in range(numberOfObjectives - (i + 1)):
        f[i] *= np.cos(x[j] * 0.5 * np.pi)

      if i != 0:
        aux = numberOfObjectives - (i + 1)
        f[i] *= np.sin(x[aux] * 0.5 * np.pi)

    for i in range(numberOfObjectives):
      solution.objectives[i] = f[i]

    return solution