#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 16:04:10 2025

@author: gustaaragao
"""

from typing import List
from src.Solution import Solution
from src.MOEAs.crossovers.Crossover import Crossover
import numpy as np

class TwoPointCrossover(Crossover):
    def __init__(self, distributionIndex, crossoverProbability):
        super().__init__(distributionIndex, crossoverProbability)
    
    def crossover(self, solutions: List[Solution]):
        solution1 = solutions[0]
        solution2 = solutions[1]

        if self.crossoverProbability > np.random.uniform(0.0, 1.0):
            return [solution1, solution2]
        
        v = solution1.decisionVariables.copy()
        w = solution2.decisionVariables.copy()

        l = len(v)

        c = np.random.randint(0, l)
        d = np.random.randint(0, l)

        if c > d:
            c, d = d, c

        if c != d:
            for i in range(c, d): # Swap on (0, l - 1)
                v[i], w[i] = w[i], v[i]
        
        offspring1 = Solution(solution1.problem)
        offspring2 = Solution(solution2.problem)
        
        offspring1.decisionVariables = v
        offspring2.decisionVariables = w
        
        return [offspring1, offspring2]