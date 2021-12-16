#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:56:56 2021

@author: jad
"""

from src.MOEAs.Algorithm import Algorithm

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
      for individual in self.population:
        self.problem.evaluate(individual)
    self.createOffspring()

    while self.evaluations < self.maxEvaluations:
      mixedPopulation = self.population.union(self.offspring)
      self.population.clear()
      self.offspring.clear()
      
      self.paretoFront.fastNonDominatedSort(list(mixedPopulation))

      for f in self.paretoFront.getInstance().front:
        for solution in f:
          if len(self.population) < self.populationSize:
            self.population.add(solution.clone())
      
      for i in range(int(self.populationSize/2)):
        self.evolute()
