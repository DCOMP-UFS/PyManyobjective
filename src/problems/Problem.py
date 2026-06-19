# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:47:19 2020

@author: jadso
"""

from src.Solution import Solution
from numpy import random

# Classe abstrata dos problemas
class Problem(object):
  # Construtor
  def __init__(self, numberOfObjectives,
               numberOfDecisionVariables,
               decisionVariablesLimit=None):
    self.decisionVariablesLimit    = list()
    self.numberOfDecisionVariables = numberOfDecisionVariables
    self.numberOfObjectives        = numberOfObjectives
    self.avaliations               = 0
    self.evaluation_time           = 0.0
    
    if not decisionVariablesLimit is None:
      for i in decisionVariablesLimit:
        self.decisionVariablesLimit.append(i)
        
    # Wrap self.evaluate to automatically track objective function evaluations and time
    original_evaluate = self.evaluate
    def wrapped_evaluate(solution):
      import time
      self.avaliations += 1
      start = time.perf_counter()
      res = original_evaluate(solution)
      self.evaluation_time += time.perf_counter() - start
      return res
    self.evaluate = wrapped_evaluate
    
  # Metódos concretos
  
  def getNextSolution(self, solution, start, end, increment):
    valVarVez = solution.decisionVariable[self.varVez]
    valVarVez += increment
    
    while(valVarVez >= 1):
      valVarVez = 0.0
      solution.decisionVariable(self.varVez, valVarVez)
      
      self.varVez -= 1
      if self.varVez < start:
        return False
      
      valVarVez = solution.decisionVariable[self.varVez]
        
    if self.varVez != end:
      valVarVez += increment
      
    valVarVez = min(1.0, valVarVez)
    
    solution.decisionVariable[self.varVez] = valVarVez
    self.varVez = end
    
    return True
  
  def generateSolution(self):
    solution = Solution(numberOfObjectives=self.numberOfObjectives,
                        numberOfDecisionVariables=self.numberOfDecisionVariables)
    
    for i in range(self.numberOfDecisionVariables):
      lower = self.decisionVariablesLimit[0][i]
      upper = self.decisionVariablesLimit[1][i]
      solution.decisionVariables[i] = random.randint(low=lower,high=int(upper*10000))/(upper*10000)
      
    return solution
  
  # Métodos abstratos
  def evaluate(self, solution: Solution):
    raise NotImplementedError
  
  def evaluateConstraints(self, solution: Solution):
    raise NotImplementedError

  def generateParetoFront(self):
    raise NotImplementedError