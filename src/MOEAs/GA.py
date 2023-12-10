#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:56:56 2021

@author: jad
"""

from src.MOEAs.Algorithm import Algorithm
import numpy as np
from src.Population import Population

# Classe do algoritmo NSGA-II
class GA(Algorithm):
  def __init__(self,
               problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    super(GA, self).__init__(problem,
                                maxEvaluations,
                                populationSize,
                                offSpringPopulationSize,
                                crossover,
                                mutation,
                                selection,
                                sparsity)
    
  def execute(self, initialPopulation=None):
    if initialPopulation == None:
      self.initializePopulation()
    else:
      self.population = initialPopulation

    self.problem.evaluate(self.population)

    while self.evaluations < self.maxEvaluations:
      if (self.evaluations % 1000) == 0:
        print("Evaluations: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")

      self.createOffspring()

      self.population.join(self.offspring)

      fronts = self.paretoFront.fastNonDominatedSort(self.population)
      fronts_order = np.argsort(fronts)
      
      self.population.filter(fronts_order)
      self.population.shrink(self.populationSize)
