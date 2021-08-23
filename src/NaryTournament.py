#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:24 2021

@author: jad
"""

from src.Selection import Selection
from numpy import random

# Classe de torneio binário
class NaryTournament(Selection):
  def __init__(self, tournamentSize):
    super(NaryTournament, self)
    self.tournamentSize = tournamentSize
    
  def select(self, solutions):
    if len(solutions) == 0:
      return None
    
    selectedSolution = solutions[0]
    
    if len(solutions) > 1:
      selectedSolutions = random.choice(solutions, self.tournamentSize)
    
      selectedSolution = sorted(selectedSolutions, key=lambda x:x.rank)[0]
      
    return selectedSolution