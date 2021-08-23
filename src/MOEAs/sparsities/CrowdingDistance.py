#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:13 2021

@author: jad
"""

from src.MOEAs.sparsities.Sparsity import Sparsity

# Classe de crowding distance
class CrowdingDistance(Sparsity):
  def __init__(self):
    super(CrowdingDistance, self)
    
  #  Computa as distancias de cada solução das fronteiras e salva no atributo
  # sparsity
  def compute(self, population):
    populationSize = len(population)
    
    if populationSize == 0:
      return None
    elif populationSize == 1:
      population[0].sparsity = 1e10
      return population
    elif populationSize == 2:
      population[0].sparsity = 1e10
      population[1].sparsity = 1e10
      return population
    
    front = list()
    for solution in population:
      solution.sparsity = 0.0
      front.append(solution)
    
    numberOfObjectives = population[0].numberOfObjectives
    
    for i in range(numberOfObjectives):
      front.sort(key=lambda x: x.objectives[i])
      objectiveMin = front[0].objectives[i]
      objectiveMax = front[populationSize-1].objectives[i]
      
      front[0].sparsity                = 1e10
      front[populationSize-1].sparsity = 1e10
      
      for j in range(1, populationSize-1):
        distance = front[j+1].objectives[i] - front[j-1].objectives[i]
        distance = distance / (objectiveMax - objectiveMin)
        distance += front[j].sparsity
        front[j].sparsity = distance
        
    return front