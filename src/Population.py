import numpy as np

from scipy.spatial.distance import cdist

from pyDOE2 import lhs

class Population:
    def __init__(self, numberOfObjectives, numberOfDecisionVariables):
        self.numberOfObjectives = numberOfObjectives
        self.numberOfDecisionVariables = numberOfDecisionVariables

        self.__decisionVariables = np.zeros((0, numberOfDecisionVariables))
        self.__objectives = np.zeros((0, numberOfObjectives))
        
        self.cluster = None
    
    @property
    def decisionVariables(self):
        return self.__decisionVariables
    
    @decisionVariables.setter
    def decisionVariables(self, X):
        assert(X.shape[1] == self.numberOfDecisionVariables)
        self.__decisionVariables = X
    
    @property
    def objectives(self):
        return self.__objectives
    
    @objectives.setter
    def objectives(self, Y):
        if (Y.shape[1] != self.numberOfObjectives):
            print("oh oh: ", Y.shape)
            print(Y)
        assert(Y.shape[0] == self.__decisionVariables.shape[0])
        assert(Y.shape[1] == self.numberOfObjectives)
        self.__objectives = Y
    
    def join(self, o):
        assert(o.decisionVariables.shape[1] == self.numberOfDecisionVariables)

        self.__decisionVariables = np.concatenate((self.__decisionVariables, o.decisionVariables))
        self.__decisionVariables, unique_indexes = np.unique(self.__decisionVariables, axis=0, return_index=True)
        if (self.__decisionVariables.shape[0] == 0 and o.objectives.shape[0] > 0) or (self.__objectives.shape[0] > 0 and o.objectives.shape[0] > 0) and (o.objectives.shape[1] == self.numberOfObjectives):
            self.__objectives = np.concatenate((self.__objectives, o.objectives))[unique_indexes]
        else:
            self.__objectives = np.zeros((0, self.numberOfObjectives))
    
    def clone(self):
        p = Population(self.numberOfObjectives, self.numberOfDecisionVariables)
        p.decisionVariables = self.decisionVariables
        p.objectives = self.objectives
        return p

def genPopulation(problem, samples):
    decisionVariables = lhs(problem.numberOfDecisionVariables, samples=samples)
    decisionVariables *= (np.array(problem.decisionVariablesLimit[1]) - np.array(problem.decisionVariablesLimit[0]))
    decisionVariables += np.array(problem.decisionVariablesLimit[0])

    population = Population(problem.numberOfObjectives, problem.numberOfDecisionVariables)
    population.decisionVariables = decisionVariables
    
    return population
