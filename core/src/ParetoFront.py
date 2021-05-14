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
  
  def getFront(self, index):
    if index < self.size():
      return self.getInstance().front[index]
    else:
      return list()
  
  def size(self):
    return len(self.getInstance().front)
  
  def clearFront(self):
    self.getInstance().front.clear()
  
  def addAll(self, solutionList):
    self.getInstance().front.append(list())
    for solution in solutionList:
      self.getInstance().front[-1].append(solution.clone())
  
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
    
    dominationRank = [(dominateMe[index],index) for index in range(len(dominateMe))]
    dominationRank.sort()
    
    rank = 1
    i    = 1
    for i in range(len(dominationRank)):
      dom, index = dominationRank[i]
      if dominationRank[i] > dominationRank[i-1]:
        rank += 1
      if dom > 0:
        front[rank].append(index)

    self.clearFront()
    
    solutionList = list()
    for i in front[0]:
      solutionList.append(population[i].clone())
      
    self.addAll(solutionList)

    for i in range(1, len(front)):
      if len(front[i]) > 0:
        solutionList = list()
        for j in front[i]:
          solutionList.append(population[j].clone())
        self.addAll(solutionList)