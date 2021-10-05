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
    self.numberOfDecisionVariables = numberOfDecisionVariables
    self.numberOfObjectives        = numberOfObjectives
    self.avaliations               = 0
    
    if decisionVariablesLimit is None:
      raise Exception
    else:
      self.decisionVariablesLimit = decisionVariablesLimit
    
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
  def evaluate():
    pass
  
  def evaluateConstraints():
    pass

  def generateParetoFront():
    pass