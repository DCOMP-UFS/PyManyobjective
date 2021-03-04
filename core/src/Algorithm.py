# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:26:33 2021

@author: jcpereira
"""

from ParetoFront import ParetoFront
from numpy import random
from matplotlib import pyplot as plt
import numpy as np

# Classe abstrata do algoritmos
class Algorithm:
  # Construtor
  def __init__(self, problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    
    self.problem                 = problem
    self.maxEvaluations          = maxEvaluations
    self.populationSize          = populationSize
    self.offSpringPopulationSize = offSpringPopulationSize
    self.crossover               = crossover
    self.mutation                = mutation
    self.selection               = selection
    self.sparsity                = sparsity
    self.population              = set()
    self.evaluations             = 0
    self.paretoFront             = ParetoFront()
    self.offspring               = set()
    self.lowerBound              = [ np.Inf for _ in range(problem.numberOfDecisionVariables)]
    self.upperBound              = [-np.Inf for _ in range(problem.numberOfDecisionVariables)]
  
  def clonePopulation(self):
    population = set()
    for p in self.population:
      population.add(p.clone())
      
    return population
  
  def evolute(self):
    parent1 = self.selection.select(list(self.population))
    parent2 = self.selection.select(list(self.population))
    
    lower = self.problem.decisionVariablesLimit[0]
    upper = self.problem.decisionVariablesLimit[1]
    
    children = self.crossover.crossover([parent1, parent2],lower,upper)
    
    children[0] = self.mutation.mutate(children[0],lower,upper)
    children[1] = self.mutation.mutate(children[1],lower,upper)
      
    for solution in children:
      s = self.problem.evaluate(solution)
      self.offspring.add(s)
      self.evaluations += 1
      
  def initializePopulation(self):
    self.population.clear()
    solutionList = set()
    
    while len(solutionList) < self.populationSize:
      newSolution = self.problem.generateSolution()
      newSolution = self.problem.evaluate(newSolution) 
      solutionList.add(newSolution)
      self.evaluations += 1
      
    fronts = self.paretoFront.fastNonDominatedSort(list(solutionList))
    for f in fronts:
      for solution in solutionList:
        self.population.add(solution)
  
  def createOffspring(self):
    while len(self.offspring) < self.offSpringPopulationSize:
      self.evolute()
  
  def frontCSV(self):
    csv = ''
    for solution in self.paretoFront.getInstance().front:
      for variable in solution.objectives:
        csv += str(variable) + ','
      csv = csv[:len(csv)-1] + '\n'

    return csv
  
  # Classes abstratas
  def execute(self):
    pass
  

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
        print("Epoch: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
      
      mixedPopulation = self.population.union(self.offspring)
      self.population.clear()
      self.offspring.clear()
      
      fronts = self.paretoFront.fastNonDominatedSort(list(mixedPopulation))
      
      for front in fronts:
        ordered_front = self.sparsity.compute(front)
        ordered_front = sorted(ordered_front, key=lambda x: x.sparsity)
        
        
        for solution in ordered_front:
          if len(self.population) < self.populationSize:
            self.population.add(solution.clone())
      
      for i in range(int(self.populationSize/2)):
        self.evolute()
    