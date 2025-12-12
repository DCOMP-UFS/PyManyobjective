#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 15:26:43 2025

@author: gustaaragao
"""

import numpy as np
from src.MOEAs.mutations.Mutation import Mutation
from src.Solution import Solution

class BoundedUniformConvolutionMutation(Mutation):
    """Algorithm 8 - Bounded Uniform Convolution"""

    def __init__(self, mutationProbability, range_noise):
        """
        Args:
            mutationProbability: probability of adding noise to an element in the vector
            range_noise: half-range of uniform noise [-r, r]
        """
        super().__init__(mutationProbability=mutationProbability, distributionIndex=None)

        self.range_noise = range_noise


    def mutate(self, individual: Solution, lowerBound, upperBound):
        # Apply bounded uniform convolution mutation to the individual.
        for i in range(individual.numberOfDecisionVariables):
            if self.mutationProbability >= np.random.uniform(0.0, 1.0):
                valid = False
                while not valid:
                    n = np.random.uniform(-self.range_noise, self.range_noise)
                    
                    new_value = individual.decisionVariables[i] + n
                    if lowerBound[i] <= new_value <= upperBound[i]:
                        valid = True
                
                individual.decisionVariables[i] = new_value
        
        return individual

    # Check if value is within bounds and adjust if necessary.
    def checkBounds(self, value, lower, upper):
        if value < lower:
            return lower
        elif value > upper:
            return upper
        return value