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
    
  def execute(self, initialPopulation=None):
    if initialPopulation == None:
      self.initializePopulation()
    else:
      self.population = initialPopulation
      
    self.problem.evaluate(self.population)
    self.createOffspring()

    while self.evaluations < self.maxEvaluations:
      if (self.evaluations % 1000) == 0:
        print("Evaluations: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
      
      self.population.join(self.offspring)

      fronts = self.paretoFront.fastNonDominatedSort(self.population)
      fronts_order = np.argsort(fronts)

      self.population.decisionVariables = self.population.decisionVariables[fronts_order]
      self.population.objectives = self.population.objectives[fronts_order]

      last = fronts == fronts[fronts_order][:self.populationSize][-1]
      last_population = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
      last_population.decisionVariables = self.population.decisionVariables[last]
      last_population.objectives = self.population.objectives[last]

      crowding_distance = self.sparsity.compute(last_population) * -1
      last_order = np.argsort(crowding_distance)
      self.population.decisionVariables[last] = self.population.decisionVariables[last][last_order]
      self.population.objectives[last] = self.population.objectives[last][last_order]

      self.population.decisionVariables = self.population.decisionVariables[:self.populationSize]
      self.population.objectives = self.population.objectives[:self.populationSize]
      
      self.offspring = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
      self.evolute()
