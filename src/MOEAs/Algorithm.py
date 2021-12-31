# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:26:33 2021

@author: jcpereira
"""
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from src.ParetoFront import ParetoFront
from src.Population import Population, genPopulation
import numpy as np
import sys

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
  
  def clonePopulation(self):
    population = set()
    for p in self.population:
      population.add(p.clone())
      
    return population
  
  def evolute(self, evaluate=True):
    parents1 = self.selection.select(self.population, self.problem)
    parents2 = self.selection.select(self.population, self.problem)
    
    lower = np.array(self.problem.decisionVariablesLimit[0])
    upper = np.array(self.problem.decisionVariablesLimit[1])
    
    children1, children2 = self.crossover.crossover(parents1, parents2, lower, upper)
    children = children1
    children.join(children2)

    self.mutation.mutate(children, lower, upper)
    
    self.offspring.join(children)

    if evaluate:
      self.problem.evaluate(self.offspring)

    self.evaluations += parents1.decisionVariables.shape[0] + parents2.decisionVariables.shape[0]
      
  def initializePopulation(self):
    self.population = genPopulation(self.problem, self.populationSize)
  
  def createOffspring(self):
    self.offspring = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
    self.offspring.decisionVariables = np.zeros((0, self.problem.numberOfDecisionVariables))
    while self.offspring.decisionVariables.shape[0] < self.offSpringPopulationSize:
      self.evolute(evaluate=False)
    self.problem.evaluate(self.offspring)
  
  # Classes abstratas
  def execute(self):
    pass