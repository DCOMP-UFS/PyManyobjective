#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 12:10:04 2025

@author: gustaaragao
"""

from src.problems.Problem import Problem
from src.Solution import Solution
from math import cos, exp, sqrt
import numpy as np

class Ackley(Problem):
    def __init__(self, numberOfDecisionVariables, a, b, c):
        numberOfObjectives = 1
        
        lowerBounds = [-100.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [100.0 for _ in range(numberOfDecisionVariables)]
        
        self.a = a
        self.b = b
        self.c = c
        
        super(Ackley, self).__init__(numberOfObjectives, 
                               numberOfDecisionVariables, 
                               (lowerBounds, upperBounds))
        
        self.problem = "ackley"
        
    def ideal_point(self):
        return np.array([0.0])
    
    def evaluate(self, solution: Solution):
        x = solution.decisionVariables
        # Parameters
        D = self.numberOfDecisionVariables
        f = -self.a * exp(-self.b * sqrt((1 / D) * sum([xi**2 for xi in x]))) - exp((1 / D) * sum([cos(self.c*xi) for xi in x])) + self.a + exp(1)
        
        solution.objectives[0] = f
        self.avaliations += 1
        
        return solution