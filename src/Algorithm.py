# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:26:33 2021

@author: jcpereira
"""
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from ParetoFront import ParetoFront
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
    self.offSpringPopulationSize = int(populationSize/2)
    self.crossover               = crossover
    self.mutation                = mutation
    self.selection               = selection
    self.sparsity                = sparsity
    self.population              = set()
    self.evaluations             = 1
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
    parent1 = self.selection.select(list(self.population.copy()))
    parent2 = self.selection.select(list(self.population.copy()))
    
    lower = self.problem.decisionVariablesLimit[0]
    upper = self.problem.decisionVariablesLimit[1]
    
    children = self.crossover.crossover([parent1, parent2],lower,upper)
    
    children[0] = self.mutation.mutate(children[0],lower,upper)
    children[1] = self.mutation.mutate(children[1],lower,upper)
    
    for solution in children:
      s = self.problem.evaluate(solution.clone())
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
      
    self.paretoFront.fastNonDominatedSort(list(solutionList))
    for f in self.paretoFront.getInstance().front:
      for solution in f:
        self.population.add(solution)
    
  
  def createOffspring(self):
    self.offspring.clear()
    while len(self.offspring) < self.offSpringPopulationSize:
      self.evolute()
  
  # Classes abstratas
  def execute(self):
    pass        