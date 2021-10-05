#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:25 2021

@author: jad
"""

from numpy import random
import numpy as np
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
    
  def mutate(self, individual, lowerBound, upperBound):
    for i in range(individual.numberOfDecisionVariables):
      if random.random() <= self.mutationProbability:
        y = individual.decisionVariables[i]
        yl = lowerBound[i]
        yu = upperBound[i]
        
        if yl == yu:
          y = yl
        else:
          delta1 = (y - yl) / (yu - yl)
          delta2 = (yu - y) / (yu - yl)
          deltaq = 0.0
          rand = random.random()
          mutPow = 1.0 / (self.distributionIndex + 1.0)
          
          if rand < 0.5:
            xy = 1.0 - delta1
            
            val = 2.0 * rand
            val += (1.0 - 2.0 * rand) * xy ** (self.distributionIndex + 1.0)
              
            deltaq = val ** mutPow - 1.0
          else:
            xy = 1.0 - delta2
            
            val = 2.0 * (1.0 - rand)
            val += 2.0 * (rand - 0.5) * xy ** (self.distributionIndex + 1.0)
            
            deltaq = 1.0 - val ** mutPow
          
          y = y + deltaq * (yu - yl)
          y = self.checkBounds(y, yl, yu)
          
        individual.decisionVariables[i] = y
      
      return individual