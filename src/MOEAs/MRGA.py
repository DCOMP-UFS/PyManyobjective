#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.MOEAs.Algorithm import Algorithm
import numpy as np

# Classe do algoritmo NSGA-II


class MRGA(Algorithm):
    def __init__(self, problem,
                 maxEvaluations,
                 populationSize,
                 offSpringPopulationSize,
                 crossover,
                 mutation,
                 selection,
                 sparsity,
                 R):
        super(MRGA, self).__init__(problem,
                                   maxEvaluations,
                                   populationSize,
                                   offSpringPopulationSize,
                                   crossover,
                                   mutation,
                                   selection,
                                   sparsity)
        self.R = R

    def niching(self, population, archive):
        for p in population:
            selected_leader = None
            best_dist = None
            x = np.array(p.decisionVariables)
            for q in archive:
                x_archive = np.array(q.decisionVariables)
                dist = np.linalg.norm(x - x_archive)
                if best_dist == None or dist < best_dist:
                    best_dist = dist
                    selected_leader = q
            p.cluster = selected_leader.cluster

    def memo_survival_selection(self, population, archive):
        self.niching(population, archive)

        P_clusters = []

        for i in range(len(self.R)):
            P_cluster = []
            for individual in population:
                if individual.cluster == i:
                    P_cluster.append(individual)
            P_clusters.append(sorted(P_cluster, key=lambda x: x.objectives[0]))

        selected_P = []
        ptr_cluster = [0] * len(self.R)
        k = 1
        while k <= self.populationSize:
            for i in range(len(self.R)):
                if ptr_cluster[i] >= len(P_clusters[i]):
                    continue
                selected_P.append(P_clusters[i][ptr_cluster[i]])
                ptr_cluster[i] += 1
                k += 1

        return selected_P

    def memo_tournament_selection(self, population, archive):
        self.niching(population, archive)

        P_clusters = []
        for i in range(len(self.R)):
            P_cluster = []
            for individual in population:
                if individual.cluster == i:
                    P_cluster.append(individual)
            P_clusters.append(P_cluster)

        l = 1
        selected_pop = []

        available_clusters = []
        for i in range(len(P_clusters)):
            if len(P_clusters[i]) > 0:
                available_clusters.append(i)

        while l <= self.populationSize:
            for I in range(len(available_clusters)):
                i = available_clusters[I]
                j = np.random.randint(0, len(P_clusters[i]))
                if len(P_clusters[i]) == 1:
                    selected_pop.append(P_clusters[i][j])
                else:
                    k = np.random.randint(0, len(P_clusters[i]))
                    while k == j:
                        k = np.random.randint(0, len(P_clusters[i]))
                    if P_clusters[i][j].objectives[0] <= P_clusters[i][k].objectives[0]:
                        selected_pop.append(P_clusters[i][j])
                    else:
                        selected_pop.append(P_clusters[i][k])

                # second parent
                I2 = I

                if len(available_clusters) > 1:
                    while I2 == I:
                        I2 = np.random.randint(0, len(available_clusters))

                i2 = available_clusters[I2]
                j = np.random.randint(0, len(P_clusters[i2]))
                if len(P_clusters[i2]) == 1:
                    selected_pop.append(P_clusters[i2][j])
                else:
                    k = np.random.randint(0, len(P_clusters[i2]))
                    while k == j:
                        k = np.random.randint(0, len(P_clusters[i2]))
                    if P_clusters[i2][j].objectives[0] <= P_clusters[i2][k].objectives[0]:
                        selected_pop.append(P_clusters[i2][j])
                    else:
                        selected_pop.append(P_clusters[i2][k])

                l += 2
                if l > self.populationSize:
                    break

        return selected_pop

    def evolute(self, archive):
        parents = self.memo_tournament_selection(self.population, archive)
        parents1 = []
        parents2 = []
        for i in range(int(self.populationSize / 2)):
            parents1.append(parents[i])
            parents2.append(parents[i + int(self.populationSize / 2)])
    
        lower = self.problem.decisionVariablesLimit[0]
        upper = self.problem.decisionVariablesLimit[1]
    
        for i in range(int(self.populationSize / 2)):
            children = self.crossover.crossover([parents1[i], parents2[i]],lower,upper)
    
            children[0] = self.mutation.mutate(children[0],lower,upper)
            children[1] = self.mutation.mutate(children[1],lower,upper)

            for child in children:
                self.problem.evaluate(child)
                self.offspring.add(child)
                self.evaluations += 1

    def createOffspring(self, archive):
        self.offspring.clear()
        self.evolute(archive)

    def execute(self, archive):
        self.initializePopulation()
        self.createOffspring(archive)

        while self.evaluations < self.maxEvaluations:
            if (self.evaluations % 1000) == 0:
                print("Evaluations: " + str(self.evaluations) +
                      " de " + str(self.maxEvaluations) + "...")

            mixedPopulation = self.population.union(self.offspring)
            self.population.clear()
            self.offspring.clear()

            for p in mixedPopulation:
                self.problem.evaluate(p)

            selectedPopulation = self.memo_survival_selection(mixedPopulation, archive)
            for p in selectedPopulation:
                self.population.add(p)

            self.evolute(archive)

        self.niching(self.population, archive)
        P_clusters = []
        for i in range(len(self.R)):
            P_cluster = []
            for p in self.population:
                if p.cluster == i:
                    P_cluster.append(p)
            P_clusters.append(sorted(P_cluster, key=lambda x: x.objectives[0]))

        selected = []
        for i in range(len(P_clusters)):
            if len(P_clusters[i]) == 0:
                continue

            winner = P_clusters[i][0]
            for j in range(1, len(P_clusters[i])):
                if P_clusters[i][j].objectives[0] < winner.objectives[0]:
                    winner = P_clusters[i][j]
            selected.append(winner)

        self.population = set(selected)
