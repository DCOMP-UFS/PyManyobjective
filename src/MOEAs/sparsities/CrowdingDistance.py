#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:13 2021

@author: jad
"""

from src.MOEAs.sparsities.Sparsity import Sparsity
import numpy as np

# Classe de crowding distance
class CrowdingDistance(Sparsity):
  def __init__(self):
    super(CrowdingDistance, self)
    
  #  Computa as distancias de cada solução das fronteiras e salva no atributo
  # sparsity
  def compute(self, population):
    f = population.objectives

    crowding_matrix = np.zeros(f.shape)
    f_mag = np.sqrt(np.einsum('...i,...i', f, f))
    normed_f = f / f_mag[..., np.newaxis]

    for i in range(population.numberOfObjectives):
        crowding = np.zeros(f.shape[0])
        crowding[0] = 1
        crowding[f.shape[0] - 1] = 1

        sorted_f = np.sort(normed_f[:, i])
        sorted_f_idx = np.argsort(normed_f[:, i])
        crowding[1:f.shape[0] - 1] = sorted_f[2:f.shape[0]] - sorted_f[0:f.shape[0] - 2]

        re_sort_order = np.argsort(sorted_f_idx)
        sorted_crowding = crowding[re_sort_order]
        crowding_matrix[:, i] = sorted_crowding

    crowding_distances = np.sum(crowding_matrix, axis=1)
    return crowding_distances