# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:32:09 2020

@author: jadso
"""

class ParetoFront:
  DOMINATES = 1
  DOMINATED_BY = -1
  NON_DOMINATED = 0
  paretoFront = None
  front = list()

  def getInstance(self):
    if self.paretoFront is None:
      self.paretoFront = ParetoFront()
      return self.paretoFront
    else:
      return self.paretoFront

  def addSolution(self, solution):
    self.getInstance().front.append(solution)
  
  def size(self):
    return len(self.getInstance().front)
  
  def clearFront(self):
    self.getInstance().front.clear()
    
  def contains(self, solution):
    return solution in self.getInstance().front
  
  def dominance(self, s1,s2):
    count = 0
    count2 = s1.numberOfObjectives
    
    for i in range(count2):
      if s1.objectives[i] > s2.objectives[i]:
        count += 1
      else:
        if s1.objectives[i] == s2.objectives[i]:
          count2 -= 1
          
    if count == 0:
        if count2 == 0:
          return self.NON_DOMINATED
        else:
          return self.DOMINATED_BY
    else:
      if count > 0 and count < count2:
        return self.NON_DOMINATED
      else:
        return self.DOMINATES
      
        
    
  def fastNonDominatedSort(self, population):
    populationSize = len(population)
    dominateMe     = [0 for _ in range(populationSize)]
    iDominate      = [list() for _ in range(populationSize)]
    front          = [list() for _ in range(populationSize+1)]
    
    for p in range(populationSize):
      iDominate[p]  = list()
      dominateMe[p] = 0

    for p in range(populationSize-1):
      for q in range(p+1, populationSize):
        flagDominate = self.dominance(population[p], population[q])
        if flagDominate == self.DOMINATED_BY:
          iDominate[p].append(q)
          dominateMe[q] += 1
        elif flagDominate == self.DOMINATES:
          iDominate[q].append(p)
          dominateMe[p] += 1
    
    for i in range(populationSize):
      if dominateMe[i] == 0:
          front[0].append(i)
          population[i].rank = 0
    
    
    i = 0
    while len(front[i]) > 0:
      i += 1
      for p in front[i-1]:
        for q in iDominate[p]:
          dominateMe[q] -= 1
          if dominateMe[q] == 0:
            front[i].append(q)
            population[q].rank = i

    self.clearFront()
    
    for i in front[0]:
      self.addSolution(population[i].objectives)
      
    fronts = list()
    for f in front:
      if len(f) > 0:
        fronts.append([population[i] for i in f])

    return fronts