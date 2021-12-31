#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.MOEAs.Algorithm import Algorithm
import numpy as np

from src.Population import Population
from src.Population import genPopulation

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
        X = population.decisionVariables
        X_archive = archive.decisionVariables

        selected_leader = []
        for i in range(X.shape[0]):
            selected_leader.append(np.argmin(np.sum(np.abs(X[i] - X_archive) ** 2, axis=-1) ** (1./2)))

        P_cluster = []
        P_clusters = []
        for i in range(len(self.R)):
            P_clusters.append([])
        for i in range(X.shape[0]):
            P_cluster.append(archive.cluster[selected_leader[i]])
            P_clusters[P_cluster[i]].append(i)

        population.cluster = P_cluster
        return P_clusters

    def memo_survival_selection(self, population, archive):
        self.problem.evaluate(population)
        P_clusters = self.niching(population, archive)

        #print(P_clusters)

        P_clusters_sorted = []
        for i in range(len(self.R)):
            if len(P_clusters[i]) == 0:
                P_clusters_sorted.append(np.array([]))
                continue

            kktpm_cluster_inds = np.argsort(np.squeeze(population.objectives[P_clusters[i]]))
            P_clusters_sorted.append(np.array(P_clusters[i])[kktpm_cluster_inds])
            '''print("check clusters sorted:")
            print(kktpm_cluster_inds)
            print(len(P_clusters_sorted[i]), len(P_clusters[i]))'''
            assert(len(P_clusters_sorted[i]) == len(P_clusters[i]))

        selected_P = []
        ptr_cluster = [0] * len(self.R)
        k = 1

        pop_size = population.decisionVariables.shape[0]

        while k <= pop_size:
            for i in range(len(self.R)):
                if ptr_cluster[i] >= len(P_clusters[i]):
                    continue
                selected_P.append(P_clusters_sorted[i][ptr_cluster[i]])
                ptr_cluster[i] += 1
                k += 1

        survivors = Population(population.numberOfObjectives, population.numberOfDecisionVariables)
        survivors.decisionVariables = population.decisionVariables[selected_P]
        survivors.objectives = population.objectives[selected_P]
        return survivors

    def memo_tournament_selection(self, population, archive):
        P_clusters = self.niching(population, archive)
        self.problem.evaluate(population)

        l = 1
        selected_pop = []

        assert(len(P_clusters) == len(self.R))
        available_clusters = []
        for i in range(len(P_clusters)):
            if len(P_clusters[i]) > 0:
                available_clusters.append(i)
        assert(len(available_clusters) > 0)

        pop_size = population.decisionVariables.shape[0] / 2

        while l <= pop_size:
            for I in range(len(available_clusters)):
                i = available_clusters[I]
                j = np.random.randint(0, len(P_clusters[i]))
                if len(P_clusters[i]) == 1:
                    selected_pop.append(P_clusters[i][j])
                else:
                    k = np.random.randint(0, len(P_clusters[i]))
                    while k == j:
                        k = np.random.randint(0, len(P_clusters[i]))
                    if np.sum(population.objectives[j]) <= np.sum(population.objectives[k]):
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
                    if np.sum(population.objectives[j]) <= np.sum(population.objectives[k]):
                        selected_pop.append(P_clusters[i2][j])
                    else:
                        selected_pop.append(P_clusters[i2][k])

                l += 2
                if l > pop_size:
                    break

        selected = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        selected.decisionVariables = self.population.decisionVariables[selected_pop]

        return selected

    def evolute(self, archive):
        parents1 = self.memo_tournament_selection(self.population, archive)
        parents2 = self.memo_tournament_selection(self.population, archive)

        lower = np.array(self.problem.decisionVariablesLimit[0])
        upper = np.array(self.problem.decisionVariablesLimit[1])
        
        children1, children2 = self.crossover.crossover(parents1, parents2, lower, upper)
        children = children1
        children.join(children2)

        self.mutation.mutate(children, lower, upper)
        
        self.offspring.join(children)

        self.problem.evaluate(self.offspring)

        self.evaluations += parents1.decisionVariables.shape[0] + parents2.decisionVariables.shape[0]

    def createOffspring(self, archive):
        self.offspring = Population(self.problem.numberOfObjectives, self.problem.numberOfDecisionVariables)
        self.offspring.decisionVariables = np.zeros((0, self.problem.numberOfDecisionVariables))
        while self.offspring.decisionVariables.shape[0] < self.offSpringPopulationSize:
            self.evolute(archive)
        self.problem.evaluate(self.offspring)

    def execute(self, archive):
        self.initializePopulation()
        self.createOffspring(archive)

        while self.evaluations < self.maxEvaluations:
            if (self.evaluations % 1000) == 0:
                print("Evaluations: " + str(self.evaluations) +
                      " de " + str(self.maxEvaluations) + "...")

            self.population.join(self.offspring)
            survivors = self.memo_survival_selection(self.population, archive)
            self.population.join(survivors)

            self.evolute(archive)

        self.problem.evaluate(self.population)
        P_clusters = self.niching(self.population, archive)

        winners = []
        for i in range(len(P_clusters)):
            if len(P_clusters[i]) == 0:
                continue

            winner = P_clusters[i][0]
            for j in range(1, len(P_clusters[i])):
                if self.population.objectives[P_clusters[i][j]] < self.population.objectives[winner]:
                    winner = P_clusters[i][j]
            winners.append(winner)

        self.population.decisionVariables = self.population.decisionVariables[winners]
        self.population.objectives = self.population.objectives[winners]

