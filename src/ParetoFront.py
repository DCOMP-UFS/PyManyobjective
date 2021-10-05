# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:32:09 2020

@author: jadso
"""


class ParetoFront:
  W =  1
  L = -1
  D =  0
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
    self.getInstance().front.append(solutionList)
  
  def dominance(self, s1, s2):
    count1 = 0
    count2 = 0
    
    for i in range(s1.numberOfObjectives):
      if s1.objectives[i] < s2.objectives[i]:
        count1 += 1
      elif s2.objectives[i] < s1.objectives[i]:
        count2 += 1
        break
          
    if count1 > 0 and count2 == 0:
      return self.W
    elif count1 == 0 and count2 > 0:
      return self.L
    return self.D
    
  def fastNonDominatedSort(self, population):
    populationSize = len(population)
    dominateMe     = [0 for _ in range(populationSize)]
    iDominate      = [list() for _ in range(populationSize)]
    front          = [list() for _ in range(populationSize + 1)]
    
    for p in range(populationSize):
      iDominate[p]  = list()
      dominateMe[p] = 0

    for p in range(populationSize - 1):
      for q in range(p + 1, populationSize):
        flagDominate = self.dominance(population[p], population[q])
        if flagDominate == self.W:
          iDominate[p].append(q)
          dominateMe[q] += 1
        elif flagDominate == self.L:
          iDominate[q].append(p)
          dominateMe[p] += 1
      
    for p in range(populationSize):
      if dominateMe[p] == 0:
        front[0].append(p)
    
    rank = 0
    while len(front[rank]) > 0:
      for p in front[rank]:
        for q in iDominate[p]:
          dominateMe[q] -= 1
          if dominateMe[q] == 0:
            front[rank + 1].append(q)
      rank += 1

    self.clearFront()

    for i in range(len(front)):
      if len(front[i]) > 0:
        solutionList = list()
        for j in front[i]:
          solutionList.append(population[j].clone())
        self.addAll(solutionList)