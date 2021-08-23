#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:21 2021

@author: jad
"""

from src.MOEAs.crossovers.Crossover import Crossover
from numpy import random
import numpy as np

# Classe de SBXCrossover
class SBXCrossover(Crossover):
  def __init__(self, distributionIndex, crossoverProbability):
    super(SBXCrossover, self).__init__(distributionIndex, crossoverProbability)
    
  def crossover(self, solutions, lowerBound, upperBound):
    solution1 = solutions[0]
    solution2 = solutions[1]
    
    offspring = []
    offspring.append(solution1.clone())
    offspring.append(solution2.clone())
    
    y1 = -1
    y2 = -1
    betaq = -1
    
    rnd = random.random(size=1)[0]
    if rnd <= self.crossoverProbability:
      for i in range(solution1.numberOfDecisionVariables):
        valueX1 = solution1.decisionVariables[i]
        valueX2 = solution2.decisionVariables[i]
        
        rnd = random.random(size=1)[0]
        if rnd <= 0.5:
          if np.abs(valueX1 - valueX2) > self.EPS:
            y1 = min(valueX1, valueX2)
            y2 = max(valueX1, valueX2)
            
            rand = random.random(size=1)[0]
            beta = 1.0 + (2.0 * (y1 - lowerBound[i]) / (y2 - y1))
            alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))
            
            if rand <= (1.0 / alpha):
              betaq = np.power(rand * alpha, (1.0 / (self.distributionIndex + 1.0)))
            else:
              betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / (self.distributionIndex + 1.0)))
            
            c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
            
            beta = 1.0 + (2.0 * (upperBound[i] - y2) / (y2 - y1))
            alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))
            
            if rand <= (1.0 / alpha):
              betaq = np.power(rand * alpha, (1.0 / (self.distributionIndex + 1.0)))
            else:
              betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / (self.distributionIndex + 1.0)))
              
            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
            
            # Verifica se está dentro dos limites.
            c1 = min(c1, upperBound[i])
            c2 = min(c2, upperBound[i])
            
            c1 = max(c1, lowerBound[i])
            c2 = max(c2, lowerBound[i])
            
            rnd = random.random(size=1)[0]
            if rnd <= 0.5:
              offspring[0].decisionVariables[i] = c2
              offspring[1].decisionVariables[i] = c1
            else:
              offspring[0].decisionVariables[i] = c1
              offspring[1].decisionVariables[i] = c2
          else:
            offspring[0].decisionVariables[i] = valueX1
            offspring[1].decisionVariables[i] = valueX2
        else:
          offspring[0].decisionVariables[i] = valueX2
          offspring[1].decisionVariables[i] = valueX1
    
    return offspring  
