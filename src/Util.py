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
  distance = 0
  for i in range(len(a)):
    distance += math.pow(a[i] - b[i], 2.0)
  return math.sqrt(distance)

def distanceToClosestPoint(point, front, distance):
  minDistance = np.Inf
  for i in range(len(front)):
    d = distance(point, front[i].objectives)
    if d < minDistance:
      minDistance = d
  return minDistance
		
def perpendicularDistance(direction, point):
  numerator = 0
  denominator = 0
  
  for i in range(len(direction)):
    numerator   += direction[i]*point[i]
    denominator += math.pow(direction[i], 2.0)

  k = numerator/denominator
  
  d = 0
  for i in range(len(direction)):
    d += math.pow(k*direction[i] - point[i],2.0)
    
  return math.sqrt(d)