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
        populationSize,
        maxEvaluations, 
        crossover: Crossover, 
        mutation: Mutation, 
        selection: Selection
    ):
        super(GeneticAlgorithm, self).__init__(
            problem=problem, 
            maxEvaluations=maxEvaluations, 
            populationSize=populationSize, 
            offSpringPopulationSize=populationSize, 
            crossover=crossover, 
            mutation=mutation, 
            selection=selection, 
            sparsity=None
        )

    def execute(self, initialPopulation=None):
        """Algorithm 20 - The Genetic Algorithm (GA) for single-objective minimization."""

        if self.populationSize % 2 != 0:
            raise ValueError("populationSize must be even (Algorithm 20 requirement)")

        lower = self.problem.decisionVariablesLimit[0]
        upper = self.problem.decisionVariablesLimit[1]

        # Initialize P
        if initialPopulation is None:
            population = [self.problem.generateSolution() for _ in range(self.populationSize)]
        else:
            population = [p.clone() for p in list(initialPopulation)]
            if len(population) != self.populationSize:
                raise ValueError("initialPopulation size must match populationSize")

        # Evaluate initial population
        self.evaluations = 0
        for i in range(len(population)):
            population[i] = self.problem.evaluate(population[i])
            population[i].evaluated = True
            self.evaluations += 1

        best: Solution = None

        # Main loop
        while self.evaluations < self.maxEvaluations:
            # AssessFitness + track Best
            for individual in population:
                if not getattr(individual, "evaluated", False):
                    self.problem.evaluate(individual)
                    individual.evaluated = True
                    self.evaluations += 1

                if best is None or individual.objectives[0] < best.objectives[0]:
                    best = individual.clone()

            # Generate Q
            offspring_population = []
            for _ in range(self.populationSize // 2):
                parent_a = self.selection.select(population)
                parent_b = self.selection.select(population)

                children = self.crossover.crossover([parent_a.clone(), parent_b.clone()], lower, upper)
                children[0] = self.mutation.mutate(children[0], lower, upper)
                children[1] = self.mutation.mutate(children[1], lower, upper)

                children[0].evaluated = False
                children[1].evaluated = False
                offspring_population.extend(children)

            population = offspring_population

        # Expose final population (best effort) in the base attribute too
        self.population = set(population)
        return best