#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gustaaragao
"""

from src.Selection import Selection
import numpy as np

# -1 * FITNESS ou 1 / FITNESS na Minimização

class FitnessProportionate(Selection):
    def __init__(self):
        super().__init__()

    def select(self, solutions):
        """Algorithm 30 - Fitness-Proportionate Selection (roulette wheel).

        Notes:
        - The book assumes fitness values are >= 0 and higher is better.
        - This framework primarily uses minimization (smaller objective is better).
          Here we convert objective[0] into a non-negative fitness by linear scaling:
          fitness_i = max(obj) - obj_i.
        - If all fitness values are 0 (all objectives equal), selection is uniform.
        """
        if solutions is None or len(solutions) == 0:
            return None

        objectives = np.asarray([s.objectives[0] for s in solutions], dtype=float)
        if not np.all(np.isfinite(objectives)):
            finite = objectives[np.isfinite(objectives)]
            fallback = float(np.max(finite)) if finite.size else 0.0
            objectives = np.where(np.isfinite(objectives), objectives, fallback)

        fitness = float(np.max(objectives)) - objectives
        if np.all(fitness == 0.0):
            fitness = np.ones_like(fitness)

        cdf = np.cumsum(fitness)
        total = float(cdf[-1])
        if total <= 0.0:
            return solutions[int(np.random.randint(0, len(solutions)))]

        n = float(np.random.uniform(0.0, total))
        idx = int(np.searchsorted(cdf, n, side="right"))
        idx = min(max(idx, 0), len(solutions) - 1)
        return solutions[idx]

