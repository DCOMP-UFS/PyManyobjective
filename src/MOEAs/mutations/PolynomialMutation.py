#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:25 2021

@author: jad
"""

from numpy import random
import numpy as np
from numpy.lib.function_base import diff
from src.MOEAs.mutations.Mutation import Mutation

# Classe de mutação polinomial (conforme implementação presente no JMetal)
class PolynomialMutation(Mutation):
  def __init__(self, mutationProbability, distributionIndex):
    super(PolynomialMutation, self).__init__(mutationProbability, distributionIndex)
    
  def checkBounds(self, value, lower, upper):
    if str(value) == 'nan':
      return upper
    
    if value < lower:
      value = lower
    elif value > upper:
      value = upper
      
    return value
    
  def mutate(self, population, lowerBound, upperBound):
        X = population.decisionVariables

        y = np.copy(X)

        delta1 = (y - lowerBound) / (upperBound - lowerBound)
        delta2 = (upperBound - y) / (upperBound - lowerBound)
        deltaq = np.zeros(X.shape)

        mutPow = 1.0 / (self.distributionIndex + 1.0)

        rand = np.random.random(X.shape)
        use_delta1 = rand < 0.5
        use_delta2 = rand >= 0.5

        xy = 1.0 - delta1
        val = 2.0 * rand
        val += (1.0 - 2.0 * rand) * xy ** (self.distributionIndex + 1.0)

        deltaq[use_delta1] = val[use_delta1] ** mutPow - 1.0

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand)
        val += 2.0 * (rand - 0.5) * xy ** (self.distributionIndex + 1.0)

        deltaq[use_delta2] = 1.0 - val[use_delta2] ** mutPow

        mutation_random = np.random.random(X.shape)
        do_mutation = mutation_random <= self.mutationProbability

        bound_diff = (upperBound - lowerBound)
        bound_diff = np.tile(bound_diff, (X.shape[0], 1))

        y[do_mutation] += deltaq[do_mutation] * bound_diff[do_mutation]
        y = np.minimum(y, upperBound)
        y = np.maximum(y, lowerBound)

        population.decisionVariables = y