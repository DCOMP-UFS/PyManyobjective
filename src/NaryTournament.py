#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:24 2021

@author: jad
"""

from src.Selection import Selection
from src.Population import Population
import numpy as np

# Classe de torneio binário
class NaryTournament(Selection):
  def __init__(self, tournamentSize):
    super(NaryTournament, self)
    self.tournamentSize = tournamentSize

  def select(self, population, problem):
    A = np.random.permutation(self.tournamentSize)
    B = np.random.permutation(self.tournamentSize)

    A_solutions = population.decisionVariables[A]
    A_objectives = population.objectives[A]

    B_solutions = population.decisionVariables[A]
    B_objectives = population.objectives[B]

    best = np.minimum(A_objectives, B_objectives)
    check_A = (A_objectives == best).all(axis=1)
    check_B = (B_objectives == best).all(axis=1)
    draw = np.logical_not(np.logical_and(check_A, check_B)) # ~xor
    check_A[draw] = False
    check_B[draw] = False

    selected = Population(problem.numberOfObjectives, problem.numberOfDecisionVariables)
    selected.decisionVariables = A_solutions
    selected.objectives = A_objectives
    selected.decisionVariables[check_B] = B_solutions[check_B]
    selected.objectives[check_B] = B_objectives[check_B]

    return selected