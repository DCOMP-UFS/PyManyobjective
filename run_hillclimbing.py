#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 23:59:22 2025

@author: gustaaragao
"""

from src.problems.Sphere import Sphere
from src.MOEAs.HillClimbing import HillClimbing

# Parameters
D = 30
max_evaluations = 50_000

p_values = [0.125, 0.25, 0.5, 1.0]
r_values = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0]

def simulate(p, r):
   # Create the problem
    problem = Sphere(numberOfDecisionVariables=D)
    
    # Create the algorithm
    algorithm = HillClimbing(
        problem=problem,
        maxEvaluations=max_evaluations,
        probability=p,
        range_noise=r
    )
    
    # Execute the algorithm
    best = algorithm.execute()
    
    result = {
        'p': p,
        'r': r,
        'best_fitness': best.objectives[0],
        'evaluations': algorithm.evaluations
    }
    
    print(result) 

def main():
    for p in p_values:
        for r in r_values:
            simulate(p, r)
        
if __name__ == '__main__':
    main()