# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:57:55 2021

@author: jcpereira
"""

import math
import numpy as np

# Classe de pontos de referência
class ReferencePoint:
  def __init__(self):
    self.position         = list()
    self.memberSize       = 0
    self.potentialMembers = list()
    
  def generateReferencePoints(self, numberOfObjectives, numberOfDivisions):
    referencePoints = list()
    refPoint        = [0.0 for _ in range(numberOfObjectives)]
    self.recursiveGenerator(referencePoints, refPoint, numberOfObjectives, numberOfDivisions, numberOfDivisions, 0)
    
    return referencePoints
    
  def recursiveGenerator(self, referencePoints, refPoint, m, left, total, element):
    if element == (m - 1):
      refPoint[element] = float(left)/float(total)
      referencePoints.append(self.copy(refPoint))
    else:
      for i in range(left + 1):
        refPoint[element] = float(i)/float(total)
        self.recursiveGenerator(referencePoints, refPoint, m, left - i, total, element + 1)
        
  def copy(self, refPoint):
    newRefPoint = ReferencePoint()
    for i in range(len(refPoint)):
      newRefPoint.position.append(refPoint[i])
    return newRefPoint
    
  def addMember(self):
    self.memberSize += 1
    
  def addPotentialMember(self, member, distance):
    self.potentialMembers.append((member, distance))
    
  def sort(self):
    self.potentialMembers.sort(key=lambda x: x[1])
    
  def findClosestMember(self):
    return self.potentialMembers[0]
    
  def randomMember(self):
    import random as rand
    index = rand.choice(list(range(len(self.potentialMembers))))
    return self.potentialMembers[index]
    
  def remove(self, refPoint):
    pass

def euclideanDistance(a, b):
  # Implementação com numpy para não estourar (OverflowError) quando os
  # objetivos são muito grandes (ex.: soluções DTLZ mal convergidas com g
  # explodido). Em overflow o numpy retorna inf em vez de lançar exceção.
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  diff = a - b
  return float(np.sqrt(np.dot(diff, diff)))

def distanceToClosestPoint(point, front, distance):
  minDistance = np.Inf
  for i in range(len(front)):
    d = distance(point, front[i].objectives)
    if d < minDistance:
      minDistance = d
  return minDistance
		
def perpendicularDistance(direction, point):
  # Implementação com numpy para não estourar (OverflowError) quando o ponto
  # tem objetivos muito grandes. Em overflow o numpy retorna inf, fazendo a
  # solução simplesmente não ser escolhida como mais próxima.
  direction = np.asarray(direction, dtype=float)
  point = np.asarray(point, dtype=float)

  denominator = np.dot(direction, direction)
  if denominator == 0.0:
    return float(np.sqrt(np.dot(point, point)))

  k = np.dot(direction, point) / denominator
  diff = k * direction - point
  return float(np.sqrt(np.dot(diff, diff)))