# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:44:25 2021

@author: jcpereira
"""

# Classe abstrata de mutação
class Mutation():
  # Construtor
  def __init__(self, mutationProbability, distributionIndex):
    self.mutationProbability = mutationProbability
    self.distributionIndex = distributionIndex
  
  # Métodos abstratos
  def mutate(self, individual, lowerBound, upperBound):
    pass

  def checkBounds(self, value, lower, upper):
    pass