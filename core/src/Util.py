# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:57:55 2021

@author: jcpereira
"""

import math
from numpy import random

# Classe de pontos de referência
class ReferencePoint:
  def __init__(self):
    self.position         = list()
    self.memberSize       = 0
    self.potentialMembers = list()

  def generateReferencePoints(self, numberOfObjectives, numberOfDivisions):
    refPoint          = ReferencePoint()
    refPoint.position = [0.0 for _ in range(numberOfObjectives)]
    return self.recursiveGenerator(list(),
                                   refPoint,
                                   numberOfObjectives,
                                   numberOfDivisions,
                                   numberOfDivisions,
                                   0)
    
  
  def recursiveGenerator(self, referencePoints, refPoint, m, left, total, element):
    if element == m-1:
      refPoint.position[element] = float(left/total)
      referencePoints.append(refPoint.copy())
    else:
      for i in range(left+1):
        refPoint.position[element] = i/total
        
        self.recursiveGenerator(referencePoints,refPoint,m,left-i,total,element+1)
        
    return referencePoints
      
  def copy(self):
    referencePoint                  = ReferencePoint()
    referencePoint.position         = self.position.copy()
    referencePoint.memberSize       = self.memberSize
    referencePoint.potentialMembers = self.potentialMembers.copy()
    
    return referencePoint
  
  def addMember(self):
    self.memberSize += 1
    
  def addPotentialMember(self, member, distance):
    self.potentialMembers.append((member,distance))
    
  def sort(self):
    self.potentialMembers.sort(key=lambda x: x[1], reverse=True)
    
  def findClosestMember(self):
    return self.potentialMembers.pop(len(self.potentialMembers)-1)
  
  def randomMember(self):
    index  = random.choice([i for i in range(len(self.potentialMembers))])
    member = self.potentialMembers.pop(index)
    return member
  
  def remove(self, refPoint):
    self.potentialMembers.remove(refPoint)
  
def euclideanDistance(a, b):
  dist = 0
  for i in range(len(a)):
    dist += (a[i] - b[i])*(a[i] - b[i])
  return math.sqrt(dist)

def distanceToClosestPoint(point, front, distance):
  minDistance = math.inf
  for frontPoint in front:
    dist        = distance(point, frontPoint)
    minDistance = min(minDistance, dist)
    
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