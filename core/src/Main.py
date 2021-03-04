# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:49:41 2021

@author: jcpereira
"""

from Problem import DTLZ1
from Operator import SBXCrossover
from Operator import BinaryTournament
from Operator import PolynomialMutation
from Operator import CrowdingDistance
from Algorithm import NSGAII
from QualityIndicator import IGD
from QualityIndicator import GD

numberOfObjectives = 3
numberOfDecisionVariables = 7
k = numberOfDecisionVariables + 1 - numberOfObjectives
problem = DTLZ1(numberOfObjectives,k)

crossoverProbability = 0.9
crossoverDistributionIndex = 20.0

crossover = SBXCrossover(crossoverDistributionIndex, crossoverProbability)

mutationProbability = 1.0 / problem.numberOfDecisionVariables
mutationDistributionIndex = 20.0

mutation = PolynomialMutation(mutationProbability, mutationDistributionIndex)

selection = BinaryTournament()

sparsity = CrowdingDistance() 
populationSize = 100
maxEvaluations = 300
algorithm = NSGAII(problem=problem,
                   maxEvaluations=maxEvaluations,
                   populationSize=populationSize,
                   offSpringPopulationSize=populationSize,
                   crossover=crossover,
                   mutation=mutation,
                   selection=selection,
                   sparsity=sparsity)

algorithm.execute()

file = open('../../resources/ReferenceFronts/DTLZ/DTLZ1.3D.csv','r')
file_front = []
for line in file.read().split('\n'):
  file_front.append([float(x) for x in line.split(',') if x != ''])
  
file_front = file_front[:len(file_front)-1]
igd = IGD(file_front)
gd = GD(file_front)

print('IGD: ' + str(igd.calculate(algorithm.paretoFront.getInstance().front)))
print('GD: ' + str(gd.calculate(algorithm.paretoFront.getInstance().front)))