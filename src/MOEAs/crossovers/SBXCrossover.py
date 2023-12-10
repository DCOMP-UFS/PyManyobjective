#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:21 2021

@author: jad
"""

from src.MOEAs.crossovers.Crossover import Crossover
import numpy as np
from src.Population import Population

# Classe de SBXCrossover
class SBXCrossover(Crossover):
  def __init__(self, distributionIndex, crossoverProbability):
    super(SBXCrossover, self).__init__(distributionIndex, crossoverProbability)
    
  def crossover(self, population1, population2, lowerBound, upperBound):
        solutions1 = population1.decisionVariables
        solutions2 = population2.decisionVariables

        assert(solutions1.shape == solutions2.shape)

        children1 = np.copy(solutions1)
        children2 = np.copy(solutions2)

        random_crossover = np.random.random(solutions1.shape)
        do_nothing = random_crossover > self.crossoverProbability

        random_swap = np.random.random(solutions1.shape)
        just_swap = random_swap <= 0.5
        just_swap[do_nothing] = False

        dist = np.abs(solutions1 - solutions2)
        use_c = np.full(solutions1.shape, False)
        use_c[dist > self.EPS] = True
        use_c[do_nothing] = False
        use_c[just_swap] = False

        y1 = np.minimum(solutions1, solutions2)
        y2 = np.maximum(solutions1, solutions2)
        
        rand = np.random.random(solutions1.shape)
        beta = 1.0 + (2.0 * (y1 - lowerBound) / (y2 - y1))
        alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))

        mask = rand <= (1.0 / alpha)
        mask_not = rand > (1.0 / alpha)

        betaq = np.zeros(solutions1.shape)
        betaq[mask] = np.power(rand * alpha, (1.0 / self.distributionIndex + 1.0))[mask]
        betaq[mask_not] = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / self.distributionIndex + 1.0))[mask_not]

        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

        beta = 1.0 + (2.0 * (upperBound - y2) / (y2 - y1))
        alpha = 2.0 - np.power(beta, -(self.distributionIndex + 1.0))
        
        mask = rand <= (1.0 / alpha)
        mask_not = rand > (1.0 / alpha)

        betaq = np.zeros(solutions1.shape)
        betaq[mask] = np.power(rand * alpha, (1.0 / self.distributionIndex + 1.0))[mask]
        betaq[mask_not] = np.power(1.0 / (2.0 - rand * alpha), 1.0 / (1.0 / self.distributionIndex + 1.0))[mask_not]

        c2 = 0.5 * (y1 + y2 - betaq * (y2 - y1))

        c1 = np.minimum(c1, upperBound)
        c2 = np.minimum(c2, upperBound)

        c1 = np.maximum(c1, lowerBound)
        c2 = np.maximum(c2, lowerBound)

        shift_random = np.random.random(solutions1.shape)
        shift = shift_random <= 0.5
        tmp = np.copy(c1[shift])
        c1[shift] = c2[shift]
        c2[shift] = tmp

        children1[use_c] = c1[use_c]
        children2[use_c] = c2[use_c]

        tmp = np.copy(children1[just_swap])
        children1[just_swap] = children2[just_swap]
        children2[just_swap] = tmp

        children1_population = Population(population1.numberOfObjectives, population1.numberOfDecisionVariables)
        children1_population.decisionVariables = children1

        children2_population = Population(population2.numberOfObjectives, population2.numberOfDecisionVariables)
        children2_population.decisionVariables = children2

        return children1_population, children2_population