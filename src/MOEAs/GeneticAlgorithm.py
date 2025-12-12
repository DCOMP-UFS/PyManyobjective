#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 15:20:43 2025

@author: gustaaragao
"""

from src.MOEAs.Algorithm import Algorithm
from src.problems.Problem import Problem
from src.MOEAs.crossovers.Crossover import Crossover
from src.MOEAs.mutations.Mutation import Mutation
from src.Selection import Selection
from src.Solution import Solution
import numpy as np


class GeneticAlgorithm(Algorithm):
    def __init__(
        self, 
        problem: Problem, 
        maxEvaluations, 
        crossover: Crossover, 
        mutation: Mutation, 
        selection: Selection
    ):
        super(GeneticAlgorithm, self).__init__(
            problem=problem, 
            maxEvaluations=maxEvaluations, 
            populationSize=1, 
            offSpringPopulationSize=1, 
            crossover=crossover, 
            mutation=mutation, 
            selection=selection, 
            sparsity=None
        )

    def execute(self, initialPopulation):
        # Initialize the population
        if initialPopulation == None:
            self.initializePopulation()
        else:
            self.population = initialPopulation
        
        # Best <- {} (empty)
        best: Solution = None

        # Main loop
        self.evaluations = 1
        while self.evaluations < self.maxEvaluations:
            for individual in self.population:
                # AssessFitness(Pi)
                if not individual.evaluated: # individual is object of Solution
                    individual = self.problem.evaluate(individual)
                    self.evaluations += 1

                # best = {} or Fitness(Pi) > Fitness(Best)
                if best is None or individual.objectives[0] < best.objectives[0]:
                    best = individual.clone()
            
            # Q <- []
            offspring_population = []

            # for popsize / 2 times do
            for _ in range(self.populationSize // 2):
                parent_a = self.selection.select(self.population)
                parent_b = self.selection.select(self.population)

                # Children Ca, Cb <= Crossover(Copy(Pa), Copy(Pb))
                childrens = self.crossover.crossover(parent_a, parent_b)

                child_a = self.mutation.mutate(
                    childrens[0],
                    self.problem.decisionVariablesLimit[0],
                    self.problem.decisionVariablesLimit[1],
                )
                
                child_b = self.mutation.mutate(
                    childrens[0],
                    self.problem.decisionVariablesLimit[0],
                    self.problem.decisionVariablesLimit[1],
                )

                offspring_population.append(child_a)
                offspring_population.append(child_b)
            
            # P <- Q
            self.population.clear()
            for x in offspring_population:
                self.population.add(offspring_population)
        
        return best