# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:26:33 2021

@author: jcpereira
"""

from Util import ReferencePoint
from Util import perpendicularDistance
from ParetoFront import ParetoFront
import numpy as np
from numpy import random

# Classe abstrata do algoritmos
class Algorithm:
  # Construtor
  def __init__(self, problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    
    self.problem                 = problem
    self.maxEvaluations          = maxEvaluations
    self.populationSize          = populationSize
    self.offSpringPopulationSize = offSpringPopulationSize
    self.crossover               = crossover
    self.mutation                = mutation
    self.selection               = selection
    self.sparsity                = sparsity
    self.population              = set()
    self.evaluations             = 1
    self.paretoFront             = ParetoFront()
    self.offspring               = set()
    self.lowerBound              = [ np.Inf for _ in range(problem.numberOfDecisionVariables)]
    self.upperBound              = [-np.Inf for _ in range(problem.numberOfDecisionVariables)]
  
  def clonePopulation(self):
    population = set()
    for p in self.population:
      population.add(p.clone())
      
    return population
  
  def evolute(self):
    parent1 = self.selection.select(list(self.population))
    parent2 = self.selection.select(list(self.population))
    
    lower = self.problem.decisionVariablesLimit[0]
    upper = self.problem.decisionVariablesLimit[1]
    
    children = self.crossover.crossover([parent1, parent2],lower,upper)
    
    children[0] = self.mutation.mutate(children[0],lower,upper)
    children[1] = self.mutation.mutate(children[1],lower,upper)
      
    for solution in children:
      s = self.problem.evaluate(solution)
      self.offspring.add(s)
      self.evaluations += 1
      
  def initializePopulation(self):
    self.population.clear()
    solutionList = set()
    
    while len(solutionList) < self.populationSize:
      newSolution = self.problem.generateSolution()
      newSolution = self.problem.evaluate(newSolution) 
      solutionList.add(newSolution)
      self.evaluations += 1
      
    fronts = self.paretoFront.fastNonDominatedSort(list(solutionList))
    for f in fronts:
      for solution in solutionList:
        self.population.add(solution)
  
  def createOffspring(self):
    while len(self.offspring) < self.offSpringPopulationSize:
      self.evolute()
  
  def frontCSV(self):
    csv = ''
    for solution in self.paretoFront.getInstance().front:
      for variable in solution.objectives:
        csv += str(variable) + ','
      csv = csv[:len(csv)-1] + '\n'

    return csv
  
  # Classes abstratas
  def execute(self):
    pass
  

# Classe do algoritmo NSGA-II
class NSGAII(Algorithm):
  def __init__(self, problem,
               maxEvaluations,
               populationSize,
               offSpringPopulationSize,
               crossover,
               mutation,
               selection,
               sparsity):
    super(NSGAII,self).__init__(problem,
                                maxEvaluations,
                                populationSize,
                                offSpringPopulationSize,
                                crossover,
                                mutation,
                                selection,
                                sparsity)
    
  def execute(self):
    self.initializePopulation()
    self.createOffspring()
    
    while self.evaluations < self.maxEvaluations:
      if (self.evaluations % 1000) == 0:
        print("Epoch: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
      
      mixedPopulation = self.population.union(self.offspring)
      self.population.clear()
      self.offspring.clear()
      
      fronts = self.paretoFront.fastNonDominatedSort(list(mixedPopulation))
      
      for front in fronts:
        ordered_front = self.sparsity.compute(front)
        ordered_front = sorted(ordered_front, key=lambda x: x.sparsity)
        
        
        for solution in ordered_front:
          if len(self.population) < self.populationSize:
            self.population.add(solution.clone())
      
      for i in range(int(self.populationSize/2)):
        self.evolute()

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
    self.referencePointsTree     = dict()
    
  def translateObjectives(self, fronts):
    ideal_point = list()
    m           = self.problem.numberOfObjectives
    newFront    = set()
    for i in range(m):
      minObj = np.Inf
      for s in fronts[0]:
        minObj = min(minObj, s[i])
      
      ideal_point.append(minObj)
      
      for f in fronts:
        solutionList = list()
        for s in f:
          if i == 0:
            solutionList.append(list())
          solutionList[s.objectives[i]-minObj]
        newFront.append(solutionList)
        
    return ideal_point, newFront
        
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
  
  def findExtremePoints(self, fronts):
    extremePoints = list()
    minInd        = None
    
    for i in range(self.problem.numberOfObjectives):
      minASF = np.Inf
      
      for s in fronts[0]:
        asf = self.ASF(s, i)
        
        if asf < minASF:
          minASF = asf
          minInd = s
      extremePoints.append(minInd)
      
    return extremePoints
    
  def guassianElimination(self, A, b):
    N = len(A)
    
    for i in range(N):
      A[i].append(b[i])
      
    for base in range(N-1):
      for target in range(base+1, N):
        ratio = A[target][base]/A[base][base]
        
        for term in range(len(A[base])):
          A[target][term] = A[target][term] - A[base][term]*ratio
          
    x = [0.0 for _ in range(N)]
    
    for i in reversed(range(N)):
      for known in range(i+1, N):
        A[i][N] = A[i][N] - A[i][known]*x[known]
      x[i] = A[i][N]/A[i][i]
    
    return x

  def hyperplane(self, fronts, extremePoints):
    duplicate = False
    m         = self.problem.numberOfObjectives
    
    i = 0
    while (not duplicate) and i < len(extremePoints):
      j = i+1
      while (not duplicate) and j < len(extremePoints):
        duplicate = extremePoints[i].objectives == extremePoints[j].objectives 
        duplicate = duplicate and extremePoints[i].objectives == extremePoints[j].objectives
        
    intercepts = list()
    
    if duplicate:
      intercepts = [extremePoints[i].objectives[i] for i in range(m)]
    else:
      b = [1.0 for _ in range(m)]
      A = [s.objectives.copy() for s in extremePoints]
      x = self.guassianElimination(A,b)
      
      intercepts = [1.0/x[i] for i in range(m)]
    
    return intercepts
  
  def normalize(self, fronts, intercepts, idealPoint):
    normalizedFront = list()
    m               = self.problem.numberOfObjectives
    
    for front in fronts:
      solutionList = list()
      for s in front:
        for f in range(m):
          convObj = s.objectives
          if abs(intercepts[f]-idealPoint[f]) > 10e-10:
            convObj[f] = convObj[f]/(intercepts[f]-idealPoint[f])
          else:
            convObj[f] = convObj[f]/10e-10
        solution = s.clone()
        solution.objectives = convObj
        solutionList.append(solution)
        
      normalizedFront.append(solutionList)
    
    return normalizedFront
  
  def associate(self, fronts):
    for t in range(fronts):
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

    if not key in self.referencePointsTree.keys():
      self.referencePointsTree[key] = list()
    
    self.referencePointsTree[key] = referencePoint
      
    
  def execute(self):
    self.initializePopulation()
    self.createOffspring()
    
    while self.evaluations <= self.maxEvaluations:
      if (self.evaluations % 1000) == 0:
        print("Epoch: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")
      
      solutionSet = set()
      
      mixedPopulation = self.population.union(self.offspring)
      self.population.clear()
      self.offspring.clear()
      
      fronts = self.paretoFront.fastNonDominatedSort(list(mixedPopulation))
      
      i = 0
      while solutionSet < self.populationSize:
        ordered_front = self.sparsity.compute(fronts[i])
        ordered_front = sorted(ordered_front, key=lambda x: x.sparsity)
        
        for solution in ordered_front:
          solutionSet.add(solution.clone())
        
        i += 1
        
      front = fronts[i-1]
      if solutionSet == self.populationSize:
        for solution in solutionSet:
          self.population.add(solution)
      else:
        for solution in front:
          self.population.add(solution)
        
        k                 = self.populationSize - len(self.population)
        
        idealPoint,fronts = self.translateObjectives(fronts)
        extremePoints     = self.findExtremePoints(fronts)
        intercepts        = self.hyperplane(fronts, extremePoints)
        fronts            = self.normalize(fronts, intercepts, idealPoint)
        
        self.associate(fronts)
        
        for rp in self.referencePoints:
          rp.sort()
          self.addToTree(rp)
        
        solutionList = list()
        
        # TODO: niching
        while len(solutionList) < k:
          first         = list(self.referencePointsTree.keys())[0]
          refPointIndex = 0
          
          size = len(first)
          if size > 1:
            refPointIndex = random.choice(list(range(size)))
          
          refPoint = self.referencePointsTree[first][refPointIndex]
          self.referencePointsTree[first].remove(refPoint)
          
          if len(first) == 0:
            self.referencePoints.popitem(first)
          
          chosen = self.selectClusterMember(refPoint)
          
          
        
        
        # nicheCount   = sum([int(pi == j) for j in referenceSet])
        # solutionList = self.niching(k,nicheCount,pi,dist,referenceSet,front)

      
      for i in range(int(self.populationSize/2)):
        self.evolute()