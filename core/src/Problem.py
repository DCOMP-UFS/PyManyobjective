# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:47:19 2020

@author: jadso
"""

from Solution import Solution
from ParetoFront import ParetoFront
import numpy as np
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
    
    if not decisionVariablesLimit is None:
      for i in decisionVariablesLimit:
        self.decisionVariablesLimit.append(i)
    
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



# Problemas da classe DTLZ
class DTLZ1(Problem):
  # Construtor
  def __init__(self, numberOfObjectives=3, k=5, decisionVariablesLimit=None):
    numberOfDecisionVariables = k + numberOfObjectives - 1
    super(DTLZ1,self).__init__(numberOfObjectives,
                               numberOfDecisionVariables,
                               decisionVariablesLimit)
    self.problem = "dtlz1"
    
    lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
    upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
    
    self.decisionVariablesLimit = (lowerBounds, upperBounds)
  
  # Calcula os objetivos
  def evaluate(self, solution):
    numberOfDecisionVariables = solution.numberOfDecisionVariables
    numberOfObjectives        = solution.numberOfObjectives
    
    for i in range(numberOfObjectives):
      solution.objectives[i] = 0.0
    
    x = [0.0 for _ in range(numberOfDecisionVariables)]
    f = [0.0 for _ in range(numberOfObjectives)]
    
    k = numberOfDecisionVariables - numberOfObjectives + 1
    
    for i in range(numberOfDecisionVariables):
      x[i] = solution.decisionVariables[i]

    g = 0.0
    
    for i in range(numberOfDecisionVariables - k,
                   numberOfDecisionVariables):
      g += np.power((x[i] - 0.5), 2) - np.cos(20.0 * np.pi*(x[i] - 0.5))

    g = 100*(k + g)
    for i in range(numberOfObjectives):
      f[i] = (1.0 + g)*0.5
    
    for i in range(numberOfObjectives):
      for j in range(numberOfObjectives - (i+1)):
        f[i] *= x[j]
      
      if i != 0:
        aux = numberOfObjectives - (i+1)
        f[i] *= 1 - x[aux]
        
    for i in range(numberOfObjectives):
      solution.objectives[i] = f[i]
    
    self.avaliations += 1
    return solution
  
  # Calcula a fronteira de Pareto
  def generateParetoFront(self, population, numberOfDecisionVariables, numberOfSolutions):
    random = np.random
    random.seed(1000)
    
    pareto = ParetoFront()
    
    while(pareto.getInstance().size() < numberOfSolutions):
      numberOfObjectives        = self.numberOfObjectives
      numberOfDecisionVariables = self.numberOfDecisionVariables
      
      best = Solution(numberOfObjectives, numberOfDecisionVariables)
      
      for i in range(numberOfObjectives-1, numberOfDecisionVariables):
        best.decisionVariables[i] = 0.5
      
      for i in range(numberOfObjectives - 1):
        newVal = random.rand()
        best.decisionVariables[i] = newVal
        
      partialSum = 0.0
      best = self.evaluate(best)
      
      for i in range(best.numberOfObjectives):
        partialSum += best.objectives[i]
        
      if partialSum == 0.5:
        if not pareto.getInstance().contains(best.objectives):
          pareto.getInstance().addSolution(best.objectives)
    
    return pareto.getInstance().front
  
  def generateIncrementalParetoFront(self, numberOfDecisionVariables, increment):
    bestList = list()
    
    random = np.random
    random.seed(1000)
    
    numberOfObjectives        = self.numberOfObjectives
    numberOfDecisionVariables = self.numberOfDecisionVariables
      
    # Indicies que indicam que variaves serão geradas incrementalmente para a geracao da fronteira
 	# O padrao dos problemas DTLZ eh entre 0 e m-2
    start = 0
    end = self.numberOfObjectives - 2
    
    solution = Solution(numberOfObjectives, numberOfDecisionVariables)
    
    self.varVez = end
    
    for i in range(numberOfObjectives-1, numberOfDecisionVariables):
      solution.decisionVariables[i] = 0.5
    
    #haSolution = True
    
    objectivesList = list()
    
    while(self.getNextSolution(solution, start, end, increment)):      
      best = solution.clone()
      
      partialSum = 0.0
      best = self.evaluate(best)
      
      for objective in best.objectives:
        partialSum += objective
        
      if partialSum == 0.5:
        equals = False
        
        for objectives in objectivesList:
          equals = True
          
          for i in range(numberOfObjectives):
            if objectives[i] != best.objectives[i]:
              equals = False
              break
          
          if equals:
            break
          
        if (not equals) or (len(objectivesList) == 0):
          objectivesList.append(best.objectives)
          bestList.append(best)

    return bestList
  
# def testes():
#   numberOfObjectives = 4
# k = 5
# numberOfSolutions = 100
# ms = [4]


# for numberOfObjectives in ms:
#   print(numberOfObjectives)
#   numberOfDecisionVariables = numberOfObjectives + k - 1
#   dtlz1 = DTLZ1(numberOfObjectives, k)

#   front = dtlz1.generateParetoFront(numberOfDecisionVariables, numberOfSolutions)
#   min([min(f.objectives) for f in front])
#   max([max(f.objectives) for f in front])
# print([f.objectives[0] for f in front])
# print()
# min(front, key=lambda x: x.objectives[0])
# print([f.objectives[0] for f in front])