#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:56:56 2021

@author: jad
"""

from Algorithm import Algorithm

# Classe do algoritmo NSGA-II
class NSGAII(Algorithm):
  def __init__(self, problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    super(NSGAII,self).__init__(problem,
                                maxEvaluations,
                                populationSize,
                                offSpringPopulationSize,
                                crossover,
                                mutation,
                                selection,
                                sparsity)
    
  def execute(self):
    self.initializePopulation()
    self.createOffspring()
    
    while self.evaluations < self.maxEvaluations:
      if (self.evaluations % 1000) == 0:
        print("Evaluations: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
      
      mixedPopulation = self.population.union(self.offspring)
      self.population.clear()
      self.offspring.clear()
      
      self.paretoFront.fastNonDominatedSort(list(mixedPopulation))
      
      for f in self.paretoFront.getInstance().front:
        ordered_front = self.sparsity.compute(f)
        ordered_front = sorted(ordered_front, key=lambda x: x.sparsity)
        
        
        for solution in ordered_front:
          if len(self.population) < self.populationSize:
            self.population.add(solution.clone())
      
      for i in range(int(self.populationSize/2)):
        self.evolute()