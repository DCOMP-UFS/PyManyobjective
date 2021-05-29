# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:50:36 2020

Classe de soluções to tipo real.

@author: jadso
"""

from numpy import random

class Solution:
  # Construtor
  def __init__(self, numberOfObjectives, numberOfDecisionVariables):
    self.numberOfDecisionVariables = numberOfDecisionVariables
    self.numberOfObjectives        = numberOfObjectives
    self.decisionVariables         = list()
    self.objectives                = list()
    
    while len(self.decisionVariables) < numberOfDecisionVariables:
      self.decisionVariables.append(0.0)
    
    while len(self.objectives) < numberOfObjectives:
      self.objectives.append(0.0)
      
    self.numberOfViolatedConstraints = 0
    self.sparsity                    = -1
    self.rank                        = -1
    
  def __eq__(self, other):
    if isinstance(other, Solution):
      return self.decisionVariables == other.decisionVariables
    return False
  def __hash__(self):
    return hash(tuple(self.decisionVariables))

  def clone(self):
    solution = Solution(self.numberOfObjectives,
                        self.numberOfDecisionVariables)
    
    for i, variable in enumerate(self.decisionVariables):
      solution.decisionVariables[i] = variable
    
    for i, objective in enumerate(self.objectives):
      solution.objectives[i] = objective
      
    return solution