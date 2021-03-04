# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:44:25 2021

@author: jcpereira
"""

from numpy import random
import numpy as np

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

# Classe abstrata de seleção
class Selection():
  CROWDINGDISTANCE = 1
  RANK = 2
  
  def select(solutions):
    pass

# Classe abstrata de crossover???
class Crossover():
  # Construtor
  def __init__(self, distributionIndex, crossoverProbability):
    self.crossoverProbability = crossoverProbability
    self.distributionIndex = distributionIndex  
    self.EPS = 1.0e-14
    
  # Classes abstratas
  def crossover(self, solutions):
    pass
    

# Classe abstrata sobre populações???
class Sparsity():
  def compute(fronts):
    pass
  
# Classe de mutação polinomial (conforme implementação presente no JMetal)
class PolynomialMutation(Mutation):
  def __init__(self, mutationProbability, distributionIndex):
    super(PolynomialMutation, self).__init__(mutationProbability, distributionIndex)
    
  def checkBounds(self, value, lower, upper):
    if str(value) == 'nan':
      return upper
    
    if value < lower:
      value = lower
    elif value > upper:
      value = upper
      
    return value
    
  def mutate(self, individual, lowerBound, upperBound):
    rnd = 0.0
    delta1 = 0.0
    delta2 = 0.0
    mutPow = 0.0
    deltaq = 0.0
    y = 0.0
    yl = 0.0
    yu = 0.0
    val = 0.0
    xy = 0.0
    
    for i in range(individual.numberOfDecisionVariables):
      rand = random.randint(low=0,high=10000)/10000
      if rand <= self.mutationProbability:
        y = individual.decisionVariables[i]
        yl = lowerBound[i]
        yu = upperBound[i]
        
        if yl == yu:
          y = yl
        else:
          delta1 = (y - yl) / (yu - yl)
          delta2 = (yu - y) / (yu - yl)
          rnd = random.randint(low=0,high=10000)/10000
          mutPow = 1.0 / (self.distributionIndex + 1.0)
          
          if rnd <= 0.5:
            xy = 1.0 - delta1
            
            val = 2.0 * rnd
            val += (1.0 - 2.0*rnd)*(np.power(xy, self.distributionIndex+1.0))
              
            deltaq = np.power(val, mutPow) - 1.0
          else:
            xy = 1.0 - delta2
            
            val = 2.0 * (1.0 - rnd)
            val += 2.0*(rnd - 0.5) * (np.power(xy, self.distributionIndex+1.0))
            
            deltaq = 1.0 - np.power(val, mutPow)
          
          y = y + deltaq*(yu - yl)
          
          y = self.checkBounds(y, yl, yu)
          
        individual.decisionVariables[i] = y
      
      return individual
    
# Classe de torneio binário
class NaryTournament(Selection):
  def __init__(self, tournamentSize):
    super(BinaryTournament, self)
    self.tournamentSize = tournamentSize
    
  def select(self, solutions):
    if len(solutions) == 0:
      return None
    
    selectedSolution = solutions[0]
    
    if len(solutions) > 1:
      selectedSolutions = random.choice(solutions, self.tournamentSize)
    
      selectedSolution = sorted(selectedSolutions, key=lambda x:x.rank)[0]
      
    return selectedSolution
      
    
# Classe de torneio binário
class BinaryTournament(NaryTournament):
  def __init__(self):
    super(BinaryTournament, self).__init__(2)
    
# Classe de SBXCrossover
class SBXCrossover(Crossover):
  def __init__(self, distributionIndex, crossoverProbability):
    super(SBXCrossover, self).__init__(distributionIndex, crossoverProbability)
    
  def crossover(self, solutions, lowerBound, upperBound):
    solution1 = solutions[0]
    solution2 = solutions[1]
    
    offspring = []
    offspring.append(solution1.clone())
    offspring.append(solution2.clone())
    
    y1 = -1
    y2 = -1
    betaq = -1
    
    rnd = random.randint(low=0,high=10000)/10000
    if rnd <= self.crossoverProbability:
      for i in range(solution1.numberOfDecisionVariables):
        valueX1 = solution1.decisionVariables[i]
        valueX2 = solution2.decisionVariables[i]
        
        rnd = random.randint(low=0,high=10000)/10000
        if rnd <= 0.5:
          if np.abs(valueX1 - valueX2) > self.EPS:
            y1 = min(valueX1, valueX2)
            y2 = max(valueX1, valueX2)
            
            rand = random.randint(low=0,high=10000)/10000
            beta = 1.0 + (2.0 * (y1 - lowerBound[i]) / (y2 - y1))
            alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))
            
            if rand <= (1.0 / alpha):
              betaq = np.power(rand * alpha, (1.0 / (self.distributionIndex + 1.0)))
            else:
              betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / (self.distributionIndex + 1.0)))
            
            c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
            
            beta = 1.0 + (2.0 * (upperBound[i] - y2) / (y2 - y1))
            alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))
            
            if rand <= (1.0 / alpha):
              betaq = np.power(rand * alpha, (1.0 / (self.distributionIndex + 1.0)))
            else:
              betaq = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / (self.distributionIndex + 1.0)))
              
            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
            
            rnd = random.randint(low=0,high=10000)/10000
            if rnd <= 0.5:
              offspring[0].decisionVariables[i] = c2
              offspring[1].decisionVariables[i] = c1
            else:
              offspring[0].decisionVariables[i] = c1
              offspring[1].decisionVariables[i] = c2
          else:
            offspring[0].decisionVariables[i] = valueX1
            offspring[1].decisionVariables[i] = valueX2
        else:
          offspring[0].decisionVariables[i] = valueX2
          offspring[1].decisionVariables[i] = valueX1
    
    return offspring  

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