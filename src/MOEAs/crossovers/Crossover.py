#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:27 2021

@author: jad
"""
# Classe abstrata de crossover???
class Crossover():
  # Construtor
  def __init__(self, distributionIndex, crossoverProbability):
    self.crossoverProbability = crossoverProbability
    self.distributionIndex = distributionIndex  
    self.EPS = 1.0e-14
    
  # Classes abstratas
  def crossover(self, solutions):
    pass
    