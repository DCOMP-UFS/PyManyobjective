#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:56:58 2021

@author: jad
"""

from src.Util import ReferencePoint
from src.Util import perpendicularDistance
import numpy as np
from numpy import random
from sortedcontainers.sorteddict import SortedDict
from src.MOEAs.Algorithm import Algorithm

# Classe do algoritmo NSGA-II
class NSGAIII(Algorithm):
  
  def __init__(self, problem,
               maxEvaluations,
               crossover,
               mutation,
               selection,
               numberOfDivisions=12):
    super(NSGAIII,self).__init__(problem=problem,
                                 maxEvaluations=maxEvaluations,
                                 populationSize=0,
                                 offSpringPopulationSize=0,
                                 crossover=crossover,
                                 mutation=mutation,
                                 selection=selection,
                                 sparsity=None)
    
    self.numberOfDivisions = numberOfDivisions
    
    refPoint             = ReferencePoint()
    self.referencePoints = refPoint.generateReferencePoints(self.problem.numberOfObjectives,
                                                            self.numberOfDivisions)
    
    populationSize = len(self.referencePoints)
    while populationSize%4 > 0:
      populationSize += 1
      
    self.populationSize          = populationSize
    self.offSpringPopulationSize = populationSize
    self.referencePointsTree     = SortedDict()
  
  def ASF(self, solution, index):
    maxRatio = -np.Inf
    
    for i in range(solution.numberOfObjectives):
      w = 0
      if index == i:
        w = 1.0
      else:
        w = 0.000001
      maxRatio = max(maxRatio, solution.objectives[i]/w)
        
    return maxRatio
    
  def guassianElimination(self, A, b):
    n = len(A)
    
    for i in range(n):
      A[i].append(b[i])
    
    for base in range(n-1):
      for target in range(base+1, n):
        ratio = A[target][base]/A[base][base]
        
        for term in range(len(A[base])):
          A[target][term] = A[target][term] - A[base][term]*ratio
          
    x = [0.0 for _ in range(n)]
    
    for i in reversed(range(n)):
      for known in range(i+1, n):
        A[i][n] = A[i][n] - A[i][known]*x[known]
      x[i] = A[i][n]/A[i][i]
    
    return x

  def normalize(self, fronts):
    ideal_point   = list()
    extremePoints = list()
    m             = self.problem.numberOfObjectives
    for i in range(m):
      minObj = np.Inf
      minInd = None
      
      for s in fronts[0]:
        if s.objectives[i] < minObj:
          minObj = s.objectives[i]
          minInd = s.clone()
          
      ideal_point.append(minInd)
      
      for s in fronts[0]:
        s.objectives[i] -= minObj
    
    for i in range(m):
      minASF = np.Inf
      minInd = None
      
      for s in fronts[0]:
        asf = self.ASF(s,i)
        
        if asf < minASF:
          minASF = asf
          minInd = s.clone()

      extremePoints.append(minInd)

    duplicate = False
    
    i = 0
    while (not duplicate) and i < len(extremePoints):
      j = i+1
      while (not duplicate) and j < len(extremePoints):
        duplicate = extremePoints[i].objectives[i] == extremePoints[j].objectives[j]
        j        += 1
      i += 1
    
    intercepts = list()
    if duplicate:
      intercepts = [extremePoints[i].objectives[i] for i in range(m)]
    else:
      b = [1.0 for _ in range(m)]
      A = [s.objectives for s in extremePoints]
      x = self.guassianElimination(A,b)
      
      intercepts = [1.0/x[i] for i in range(m)]
    
    normalizedFront = list()
    for i in range(m):
      for f in fronts:
        solutionList = list()
        for s in f:
          s.objectives[i] = s.objectives[i]/intercepts[i]
          solutionList.append(s.clone())
        normalizedFront.append(solutionList)
    
    return normalizedFront
  
  def associate(self, fronts):
    for t in range(len(fronts)):
      for s in fronts[t]:
        minRefpoint = -1
        minDistance = np.Inf
        
        for r in range(len(self.referencePoints)):
          d = perpendicularDistance(self.referencePoints[r].position, s.objectives)
          
          if d < minDistance:
            minDistance = d
            minRefpoint = r

        if t+1 != len(fronts):
          self.referencePoints[minRefpoint].addMember()
        else:
          self.referencePoints[minRefpoint].addPotentialMember(s,minDistance)
  
  def selectClusterMember(self, referencePoint):
    chosen = None
    
    if len(referencePoint.potentialMembers) > 0:
      if referencePoint.memberSize == 0:
        chosen = referencePoint.findClosestMember()
      else:
        chosen = referencePoint.randomMember()
      
    return chosen
  
  
  def addToTree(self, referencePoint):
    key = referencePoint.memberSize
    if not self.referencePointsTree.__contains__(key):
      self.referencePointsTree.setdefault(key, list())
    self.referencePointsTree[key].append(referencePoint)
    
  def niching(self, fronts, front, population):
    k = self.populationSize - len(population)
    
    if k == 0:
      return population
    
    fronts = self.normalize(fronts)
    
    self.associate(fronts)
    
    for rp in self.referencePoints:
      rp.sort()
      self.addToTree(rp)
      
    solutionList = list()
    while len(solutionList) < k:
      first         = self.referencePointsTree.keys()[0]
      refPointIndex = 0
      
      size = len(self.referencePointsTree.get(first))
      if size == 1:
        refPointIndex = 0
      else:
        refPointIndex = random.choice(list(range(size)))
      
      refPoint = self.referencePointsTree.get(first).pop(refPointIndex)
      
      if len(self.referencePointsTree.get(first)) == 0:
        self.referencePointsTree.pop(first)
      
      chosen = self.selectClusterMember(refPoint)
      
      if not chosen is None:
        refPoint.addMember()
        self.addToTree(refPoint)
        solutionList.append(chosen[0])
        
    for solution in solutionList:
      population.append(solution.clone())
      
    return population
  
  def execute(self):
    self.initializePopulation()
    
    while self.evaluations <= self.maxEvaluations:
      if (self.evaluations % 1) == 0:
        print("Evaluations: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
    
      self.createOffspring()
      
      mixedPopulation = self.population.union(self.offspring.copy())
      
      self.population.clear()
      self.offspring.clear()
      
      self.paretoFront.fastNonDominatedSort(list(mixedPopulation))
      
      population = list()
      fronts     = list()
      front      = list()
      popsize    = 0
      rank       = 0
      
      while popsize < self.populationSize and rank < self.paretoFront.size():
        front = self.paretoFront.getFront(rank)
        fronts.append(front)
        popsize += len(front)
        
        if len(population) + len(front) <= self.populationSize:
          for solution in front:
            population.append(solution.clone())
        
        rank += 1
        
      if population == self.populationSize:
        for solution in population:
          self.population.add(solution.clone())
      else:
        population = self.niching(fronts.copy(), front, population)
        
        for solution in population:
          self.population.add(solution.clone())