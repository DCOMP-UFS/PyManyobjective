# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:26:33 2021

@author: jcpereira
"""
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from typing import Set
from src.Solution import Solution
from src.Selection import Selection
from src.ParetoFront import ParetoFront
from src.problems.Problem import Problem
from src.MOEAs.mutations.Mutation import Mutation
from src.MOEAs.crossovers.Crossover import Crossover

# Classe abstrata do algoritmos
class Algorithm:
  # Construtor
  def __init__(self, 
               problem: Problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    
    self.problem                    = problem
    self.maxEvaluations             = maxEvaluations
    self.populationSize             = populationSize
    # Respeita o tamanho da prole informado pelo chamador. Antes este valor era
    # sobrescrito por int(populationSize/2), ignorando o argumento recebido (o
    # run_experiments passa offSpringPopulationSize=populationSize para o
    # pure_moea). Subclasses que precisam de outro valor (ex.: NSGA-III) ainda
    # podem redefini-lo após chamar super().__init__.
    self.offSpringPopulationSize    = offSpringPopulationSize
    self.crossover: Crossover       = crossover
    self.mutation: Mutation         = mutation
    self.selection: Selection       = selection
    self.sparsity                   = sparsity
    self.population: Set[Solution]  = set()
    self.evaluations                = 0
    self.paretoFront                = ParetoFront()
    self.offspring: set[Solution]                  = set()
  
  def clonePopulation(self):
    population = set()
    for p in self.population:
      population.add(p.clone())
      
    return population
  
  def evolute(self):
    if self.evaluations >= self.maxEvaluations:
      return

    parent1 = self.selection.select(list(self.population.copy()))
    parent2 = self.selection.select(list(self.population.copy()))
    
    lower = self.problem.decisionVariablesLimit[0]
    upper = self.problem.decisionVariablesLimit[1]
    
    children = self.crossover.crossover([parent1, parent2],lower,upper)
    
    children[0] = self.mutation.mutate(children[0],lower,upper)
    children[1] = self.mutation.mutate(children[1],lower,upper)
    
    for solution in children:
      if self.evaluations >= self.maxEvaluations:
        break
      s = self.problem.evaluate(solution.clone())
      s.evaluated = True
      self.offspring.add(s)
      self.evaluations += 1
      
      
  def initializePopulation(self):
    self.population.clear()
    solutionList = set()
    
    while len(solutionList) < self.populationSize and self.evaluations < self.maxEvaluations:
      newSolution = self.problem.generateSolution()
      newSolution = self.problem.evaluate(newSolution) 
      newSolution.evaluated = True
      solutionList.add(newSolution)
      self.evaluations += 1
      
    self.paretoFront.fastNonDominatedSort(list(solutionList))
    for f in self.paretoFront.getInstance().front:
      for solution in f:
        self.population.add(solution)
    
  
  def createOffspring(self):
    self.offspring.clear()
    while len(self.offspring) < self.offSpringPopulationSize and self.evaluations < self.maxEvaluations:
      self.evolute()
  
  # Classes abstratas
  def execute(self):
    pass        
