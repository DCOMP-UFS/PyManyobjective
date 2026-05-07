from src.MOEAs.Algorithm import Algorithm
from scipy.spatial.distance import cdist
import numpy as np

class MOEAD(Algorithm):
    def __init__(self, problem,
                 maxEvaluations,
                 populationSize=None,
                 offSpringPopulationSize=1,
                 crossover=None,
                 mutation=None,
                 selection=None,
                 sparsity=None,
                 ref_dirs=None,
                 n_neighbors=20,
                 prob_neighbor_mating=0.9):
        
        if ref_dirs is None:
            from pymoo.factory import get_reference_directions
            if problem.numberOfObjectives == 2:
                ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=populationSize-1 if populationSize else 99)
            elif problem.numberOfObjectives == 3:
                ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
            else:
                ref_dirs = get_reference_directions("das-dennis", problem.numberOfObjectives, n_partitions=3)
                
        self.ref_dirs = ref_dirs
        popSize = len(self.ref_dirs)
        
        super(MOEAD, self).__init__(problem,
                                    maxEvaluations,
                                    popSize,
                                    offSpringPopulationSize,
                                    crossover,
                                    mutation,
                                    selection,
                                    sparsity)
                                    
        self.n_neighbors = min(n_neighbors, self.populationSize)
        self.prob_neighbor_mating = prob_neighbor_mating
        self.ideal_point = None
        
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

    def _tchebycheff(self, F, weights, ideal_point):
        w = np.maximum(weights, 1e-6)
        v = np.abs(F - ideal_point) * w
        return np.max(v, axis=1)

    def execute(self, initialPopulation=None):
        if initialPopulation is None:
            self.initializePopulation()
        else:
            self.population = initialPopulation
            for individual in self.population:
                if not getattr(individual, 'evaluated', False):
                    self.problem.evaluate(individual)
                    individual.evaluated = True
                
        pop_list = list(self.population)
        if len(pop_list) < self.populationSize:
            while len(pop_list) < self.populationSize:
                newSolution = self.problem.generateSolution()
                newSolution = self.problem.evaluate(newSolution)
                newSolution.evaluated = True
                pop_list.append(newSolution)
                self.evaluations += 1
        elif len(pop_list) > self.populationSize:
            pop_list = pop_list[:self.populationSize]
            
        F = np.array([ind.objectives for ind in pop_list])
        self.ideal_point = np.min(F, axis=0)
        
        while self.evaluations < self.maxEvaluations:
            if (self.evaluations % 1000) == 0:
                print("Evaluations: " + str(self.evaluations) + " de " + str(self.maxEvaluations) + "...")

            for i in np.random.permutation(len(pop_list)):
                if self.evaluations >= self.maxEvaluations:
                    break
                
                N = self.neighbors[i, :]
                
                if np.random.random() < self.prob_neighbor_mating:
                    parent_indices = np.random.permutation(N)[:2]
                else:
                    parent_indices = np.random.permutation(self.populationSize)[:2]
                
                if len(parent_indices) < 2:
                    parent_indices = [parent_indices[0], parent_indices[0]]
                    
                parent1 = pop_list[parent_indices[0]]
                parent2 = pop_list[parent_indices[1]]
                
                lower = self.problem.decisionVariablesLimit[0]
                upper = self.problem.decisionVariablesLimit[1]
                
                children = self.crossover.crossover([parent1, parent2], lower, upper)
                offspring = children[np.random.randint(0, 2)]
                offspring = self.mutation.mutate(offspring, lower, upper)
                
                s = self.problem.evaluate(offspring.clone())
                s.evaluated = True
                self.evaluations += 1
                
                self.ideal_point = np.min(np.vstack([self.ideal_point, s.objectives]), axis=0)
                
                FV = self._tchebycheff(np.array([pop_list[idx].objectives for idx in N]), self.ref_dirs[N, :], self.ideal_point)
                off_FV = self._tchebycheff(np.array([s.objectives] * len(N)), self.ref_dirs[N, :], self.ideal_point)
                
                I = np.where(off_FV < FV)[0]
                for idx in I:
                    pop_list[N[idx]] = s.clone()
                    
        self.population = set(pop_list)
        self.paretoFront.fastNonDominatedSort(pop_list)