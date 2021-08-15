#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:12 2021

@author: jad
"""
from problems.Problem import Problem
from Solution import Solution
from ParetoFront import ParetoFront
import math
import numpy as np

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

class DTLZ2(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz2"

    def evaluate(self, solution):
        numberOfDecisionVariables = self.numberOfDecisionVariables
        numberOfObjectives        = solution.numberOfObjectives

        f = [0.0 for _ in range(numberOfObjectives)]
        x = [0.0 for _ in range(numberOfDecisionVariables)]

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += (x[i] - 0.5) * (x[i] - 0.5)

        for i in range(numberOfObjectives):
            f[i] = 1.0 + g

        for i in range(numberOfObjectives):
            for j in range(numberOfObjectives - (i + 1)):
                f[i] *= np.cos(x[j] * 0.5 * np.pi)
            if i != 0:
                aux = numberOfObjectives - (i + 1)
                f[i] *= np.sin(x[aux] * 0.5 * np.pi)

        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution

class DTLZ3(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz3"

    def evaluate(self, solution):
        numberOfDecisionVariables = solution.numberOfDecisionVariables
        numberOfObjectives = solution.numberOfObjectives
        
        f = [0] * numberOfObjectives
        x = [0] * numberOfDecisionVariables

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += (x[i] - 0.5) * (x[i] - 0.5) - math.cos(20.0 * math.pi * (x[i] - 0.5))

        g = 100.0 * (k + g)
        for i in range(numberOfObjectives):
            f[i] = 1.0 + g

        for i in range(numberOfObjectives):
            for j in range(numberOfObjectives - (i + 1)):
                f[i] *= math.cos(x[j] * 0.5 * math.pi)
            if i != 0:
                aux = numberOfObjectives - (i + 1)
                f[i] *= math.sin(x[aux] * 0.5 * math.pi)
        
        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution

class DTLZ4(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz4"

    def evaluate(self, solution):
        numberOfDecisionVariables = solution.numberOfDecisionVariables
        numberOfObjectives = solution.numberOfObjectives
        
        f = [0] * numberOfObjectives
        x = [0] * numberOfDecisionVariables

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += (x[i] - 0.5) * (x[i] - 0.5)

        for i in range(numberOfObjectives):
            f[i] = 1.0 + g

        alpha = 100.0
        for i in range(numberOfObjectives):
            for j in range(numberOfObjectives - (i + 1)):
                f[i] *= math.cos(math.pow(x[j], alpha) * (math.pi / 2.0))
            if i != 0:
                aux = numberOfObjectives - (i + 1)
                f[i] *= math.sin(math.pow(x[aux], alpha) * (math.pi / 2.0))
        
        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution

class DTLZ5(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz5"

    def evaluate(self, solution):
        numberOfDecisionVariables = solution.numberOfDecisionVariables
        numberOfObjectives = solution.numberOfObjectives
        
        f = [0] * numberOfObjectives
        x = [0] * numberOfDecisionVariables

        theta = [0] * (numberOfObjectives - 1)

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += (x[i] - 0.5) * (x[i] - 0.5)

        t = math.pi / (4.0 * (1.0 + g))

        theta[0] = x[0] * math.pi / 2.0
        for i in range(1, numberOfObjectives - 1):
            theta[i] = t * (1.0 + 2.0 * g * x[i])

        for i in range(numberOfObjectives):
            f[i] = 1.0 + g

        alpha = 100.0
        for i in range(numberOfObjectives):
            for j in range(numberOfObjectives - (i + 1)):
                f[i] *= math.cos(theta[j])
            if i != 0:
                aux = numberOfObjectives - (i + 1)
                f[i] *= math.sin(theta[aux])
        
        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution

class DTLZ6(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz6"

    def evaluate(self, solution):
        numberOfDecisionVariables = solution.numberOfDecisionVariables
        numberOfObjectives = solution.numberOfObjectives
        
        f = [0] * numberOfObjectives
        x = [0] * numberOfDecisionVariables

        theta = [0] * (numberOfObjectives - 1)

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += math.pow(x[i], 0.1)

        t = math.pi / (4.0 * (1.0 + g))

        theta[0] = x[0] * math.pi / 2.0
        for i in range(1, numberOfObjectives - 1):
            theta[i] = t * (1.0 + 2.0 * g * x[i])

        for i in range(numberOfObjectives):
            f[i] = 1.0 + g

        alpha = 100.0
        for i in range(numberOfObjectives):
            for j in range(numberOfObjectives - (i + 1)):
                f[i] *= math.cos(theta[j])
            if i != 0:
                aux = numberOfObjectives - (i + 1)
                f[i] *= math.sin(theta[aux])
        
        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution

class DTLZ7(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz1"

    def evaluate(self, solution):
        numberOfDecisionVariables = self.numberOfDecisionVariables
        numberOfObjectives        = solution.numberOfObjectives

        f = [0.0 for _ in range(numberOfObjectives)]
        x = [0.0 for _ in range(numberOfDecisionVariables)]

        for i in range(numberOfDecisionVariables):
            x[i] = solution.decisionVariables[i]

        k = numberOfDecisionVariables - numberOfObjectives + 1

        g = 0.0
        for i in range(numberOfDecisionVariables - k, numberOfDecisionVariables):
            g += x[i]

        g = 1 + (9.0 * g) / k

        for i in range(numberOfObjectives - 1):
            f[i] = x[i]

        h = 0.0
        for i in range(numberOfObjectives - 1):
            h += (f[i] / (1.0 + g)) * (1 + math.sin(3.0 * math.pi * f[i]))

        h = numberOfObjectives - h

        f[numberOfObjectives - 1] = (1 + g) * h

        for i in range(numberOfObjectives):
            solution.objectives[i] = f[i]

        return solution