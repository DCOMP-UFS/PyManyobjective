#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 22:24:27 2025

@author: gustaaragao
"""

from src.problems.Problem import Problem
from src.Solution import Solution
import numpy as np

class Sphere(Problem):
    def __init__(self, numberOfDecisionVariables, decisionVariablesLimit=None):
        numberOfObjectives = 1
        
        lowerBounds = [-100.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [100.0 for _ in range(numberOfDecisionVariables)]
        
        super(Sphere, self).__init__(numberOfObjectives, 
                               numberOfDecisionVariables, 
                               (lowerBounds, upperBounds))
        
        self.problem = "sphere"
        
    def ideal_point(self):
        return np.array([0.0])
    
    def evaluate(self, solution: Solution):
        x = solution.decisionVariables
        f = sum([xi**2 for xi in x])
        
        solution.objectives[0] = f
        self.avaliations += 1
        
        return solution