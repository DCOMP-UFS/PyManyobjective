# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:49:41 2021

@author: jcpereira
"""

from DTLZ1 import DTLZ1
from SBXCrossover import SBXCrossover
from BinaryTournament import BinaryTournament
from PolynomialMutation import PolynomialMutation
from NSGAIII import NSGAIII
from QualityIndicator import IGD
from QualityIndicator import GD

numberOfObjectives = 3
k = 10
numberOfDecisionVariables = k + numberOfObjectives - 1
problem = DTLZ1(numberOfObjectives,k)

crossoverProbability = 0.9
crossoverDistributionIndex = 20.0

crossover = SBXCrossover(crossoverDistributionIndex, crossoverProbability)

mutationProbability = 1.0 / problem.numberOfDecisionVariables
mutationDistributionIndex = 20.0

mutation = PolynomialMutation(mutationProbability, mutationDistributionIndex)

selection = BinaryTournament()

maxEvaluations = 400

file = open('../resources/ReferenceFronts/DTLZ/DTLZ1.3D.csv','r')
file_front = []
for line in file.read().split('\n'):
  file_front.append([float(x) for x in line.split(',') if x != ''])
  
file_front = file_front[:len(file_front)-1]

igd_list = list()
gd_list  = list()
for i in range(1,21):
  print("Run " + str(i) + '/20...')
  algorithm = NSGAIII(problem=problem,
                      maxEvaluations=maxEvaluations,
                      crossover=crossover,
                      mutation=mutation,
                      selection=selection,
                      numberOfDivisions=12)
  
  algorithm.execute()

  igd = IGD(file_front)
  gd  = GD(file_front)
  
  front = [s.objectives for s in algorithm.paretoFront.getFront(0)]
  
  igd_list.append(igd.calculate(front))
  gd_list.append(gd.calculate(front))

import numpy as np
print('### IGD ###')
print("min: " + str(min(igd_list)))
print("max: " + str(max(igd_list)))
print("avg: " + str(np.average(igd_list)))