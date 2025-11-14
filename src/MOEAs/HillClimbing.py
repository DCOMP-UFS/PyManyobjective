#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 23:02:43 2025

@author: gustaaragao
"""

from src.MOEAs.Algorithm import Algorithm
from src.problems.Problem import Problem
from src.Solution import Solution
import numpy as np

class HillClimbing(Algorithm):
    def __init__(self, problem: Problem, maxEvaluations, probability, range_noise):
        super(HillClimbing, self).__init__(
            problem=problem, 
            maxEvaluations=maxEvaluations, 
            populationSize=1, 
            offSpringPopulationSize=1, 
            crossover=None, 
            mutation=None, 
            selection=None, 
            sparsity=None
        )
        # probability of adding noise to an element in the vector
        self.probability = probability
        # half-range of uniform noise [-r, r]
        self.range_noise = range_noise
    
    def tweak(self, solution: Solution) -> Solution:
        """Algorithm 8 - Bounded Uniform Convolution"""
        new_solution = solution.clone()
        lower = self.problem.decisionVariablesLimit[0]
        upper = self.problem.decisionVariablesLimit[1]
        
        for i in range(new_solution.numberOfDecisionVariables):
            if self.probability >= np.random.uniform(0.0, 1.0):
                valid = False
                while not valid:
                    n = np.random.uniform(-self.range_noise, self.range_noise)
                    
                    new_value = new_solution.decisionVariables[i] + n
                    if lower[i] <= new_value <= upper[i]:
                        valid = True
                
                new_solution.decisionVariables[i] = new_value
        
        return new_solution
    
    def execute(self) -> Solution:
        """Algorithm 4 - Hill Climbing"""
        # Init with a random candidate solution
        best = self.problem.generateSolution()
        best = self.problem.evaluate(best)
        self.evaluations = 1
        
        # Main loop
        while self.evaluations < self.maxEvaluations:
            # R <- Tweak(Copy(S))
            new_solution = self.tweak(best)
            new_solution = self.problem.evaluate(new_solution)
            
            # If Quality(R) > Quality(S) then S <- R
            # To minimization: if new < best, accept
            if new_solution.objectives[0] < best.objectives[0]:
                best = new_solution
            
            self.evaluations += 1
        
        self.population.clear()
        self.population.add(best)
        
        return best