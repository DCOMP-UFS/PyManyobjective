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
    
    def crossover(self, solutions: List[Solution], lowerBound=None, upperBound=None):
        solution1 = solutions[0]
        solution2 = solutions[1]

        offspring1 = solution1.clone()
        offspring2 = solution2.clone()

        rnd = float(np.random.uniform(0.0, 1.0))
        if rnd > self.crossoverProbability:
            return [offspring1, offspring2]
        
        v = offspring1.decisionVariables
        w = offspring2.decisionVariables

        l = len(v)

        c = int(np.random.randint(0, l))
        d = int(np.random.randint(0, l))

        if c > d:
            c, d = d, c

        if c != d:
            for i in range(c, d): # Swap on (0, l - 1)
                v[i], w[i] = w[i], v[i]

        return [offspring1, offspring2]