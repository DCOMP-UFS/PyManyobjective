#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:57:12 2021

@author: jad
"""
from src.problems.Problem import Problem
from src.Solution import Solution
from src.ParetoFront import ParetoFront
import math
import numpy as np

# Problemas da classe DTLZ
class DTLZ1(Problem):
  # Construtor
    def __init__(self, numberOfObjectives=3, k=5, decisionVariablesLimit=None):
        numberOfDecisionVariables = k + numberOfObjectives - 1
        self.k = k
        lowerBounds = [0.0 for _ in range(numberOfDecisionVariables)]
        upperBounds = [1.0 for _ in range(numberOfDecisionVariables)]
        decisionVariablesLimit = (lowerBounds, upperBounds)
    
        super(DTLZ1,self).__init__(numberOfObjectives,
                               numberOfDecisionVariables,
                               decisionVariablesLimit)
        self.problem = "dtlz1"

    def ideal_point(self):
        return np.array([0.0 for _ in range(self.numberOfObjectives)])
  
    def g1(self, X_M):
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.numberOfObjectives):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)

        f = np.column_stack(f)
        return f

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.numberOfObjectives - 1], x[:, self.numberOfObjectives - 1:]
        g = self.g1(X_M)

        f = []
        for i in range(0, self.numberOfObjectives):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        population.objectives = np.column_stack(f)

class DTLZ2(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz2"

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.numberOfObjectives - 1], x[:, self.numberOfObjectives - 1:]
        g = self.g2(X_M)
        
        population.objectives = self.obj_func(X_, g, alpha=1)

class DTLZ3(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz3"

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g1(X_M)

        population.objectives = self.obj_func(X_, g, alpha=1)

class DTLZ4(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz4"

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)
        
        population.objectives = self.obj_func(X_, g, alpha=self.alpha)

class DTLZ5(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz5"

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack([x[:, 0], theta[:, 1:]])

        population.objectives = self.obj_func(theta, g)

class DTLZ6(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz6"

    def evaluate(self, population):
        x = population.decisionVariables

        X_, X_M = x[:, :self.n_objs - 1], x[:, self.n_objs - 1:]
        g = np.sum(np.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack([x[:, 0], theta[:, 1:]])

        population.objectives = self.obj_func(theta, g)

class DTLZ7(DTLZ1):
    def __init__(self, numberOfObjectives=3, k=5):
        super().__init__(numberOfObjectives, k)
        self.problem = "dtlz7"

    def evaluate(self, population):
        x = population.decisionVariables

        f = []
        for i in range(0, self.n_objs - 1):
            f.append(x[:, i])
        f = np.column_stack(f)

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_objs - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)

        population.objectives = np.column_stack([f, (1 + g) * h])