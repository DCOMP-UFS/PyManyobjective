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
    rnd = 0.0
    delta1 = 0.0
    delta2 = 0.0
    mutPow = 0.0
    deltaq = 0.0
    y = 0.0
    yl = 0.0
    yu = 0.0
    val = 0.0
    xy = 0.0
    
    for i in range(individual.numberOfDecisionVariables):
      rand = random.randint(low=0,high=10000)/10000
      if rand <= self.mutationProbability:
        y = individual.decisionVariables[i]
        yl = lowerBound[i]
        yu = upperBound[i]
        
        if yl == yu:
          y = yl
        else:
          delta1 = (y - yl) / (yu - yl)
          delta2 = (yu - y) / (yu - yl)
          rnd = random.randint(low=0,high=10000)/10000
          mutPow = 1.0 / (self.distributionIndex + 1.0)
          
          if rnd <= 0.5:
            xy = 1.0 - delta1
            
            val = 2.0 * rnd
            val += (1.0 - 2.0*rnd)*(np.power(xy, self.distributionIndex+1.0))
              
            deltaq = np.power(val, mutPow) - 1.0
          else:
            xy = 1.0 - delta2
            
            val = 2.0 * (1.0 - rnd)
            val += 2.0*(rnd - 0.5) * (np.power(xy, self.distributionIndex+1.0))
            
            deltaq = 1.0 - np.power(val, mutPow)
          
          y = y + deltaq*(yu - yl)
          
          y = self.checkBounds(y, yl, yu)
          
        individual.decisionVariables[i] = y
      
      return individual