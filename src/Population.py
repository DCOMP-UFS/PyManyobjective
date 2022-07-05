import numpy as np

from scipy.spatial.distance import cdist

from pyDOE2 import lhs

class Population:
    def __init__(self, numberOfObjectives, numberOfDecisionVariables):
        self.numberOfObjectives = numberOfObjectives
        self.numberOfDecisionVariables = numberOfDecisionVariables

        self.__decisionVariables = np.zeros((0, numberOfDecisionVariables))
        self.__objectives = np.zeros((0, numberOfObjectives))

        self.evaluated = np.zeros(0, dtype=np.bool)
        
        self.cluster = None
    
    @property
    def decisionVariables(self):
        return self.__decisionVariables
    
    @decisionVariables.setter
    def decisionVariables(self, X):
        assert(X.shape[1] == self.numberOfDecisionVariables)
        self.__decisionVariables = X
        self.__objectives = np.zeros((X.shape[0], self.numberOfObjectives))
        self.evaluated = np.zeros(X.shape[0], dtype=np.bool)
    
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
        self.evaluated = np.ones(Y.shape[0], dtype=np.bool)

    def getNotEvaluatedVars(self):
        notEvaluated = ~self.evaluated
        return self.__decisionVariables[notEvaluated]

    def setNotEvaluatedObjectives(self, Y):
        notEvaluated = ~self.evaluated
        assert(Y.shape[0] == notEvaluated.sum())

        self.__objectives[notEvaluated] = Y
        self.evaluated[notEvaluated] = True
    
    def clearObjectives(self):
        self.__objectives = np.zeros(self.__objectives.shape)
        self.evaluated = np.zeros(self.evaluated.shape, dtype=np.bool)

    def join(self, o):
        assert(o.numberOfDecisionVariables == self.numberOfDecisionVariables)

        self.__decisionVariables = np.concatenate((self.__decisionVariables, o.decisionVariables))
        self.__decisionVariables, unique_indexes = np.unique(self.__decisionVariables, axis=0, return_index=True)
        self.__objectives = np.concatenate((self.__objectives, o.objectives))[unique_indexes]
        self.evaluated = np.concatenate((self.evaluated, o.evaluated))[unique_indexes]

    def add(self, individual):
        assert(individual.shape[0] == self.numberOfDecisionVariables)

        self.__decisionVariables = np.concatenate((self.__decisionVariables, np.array([individual])))
        self.__objectives = np.concatenate((self.__objectives, np.zeros((1, self.numberOfObjectives))))
        self.evaluated = np.concatenate((self.evaluated, np.zeros(1, dtype=np.bool)))

    def shrink(self, size):
        self.__decisionVariables = self.__decisionVariables[:size]
        self.__objectives = self.__objectives[:size]
        self.evaluated = self.evaluated[:size]

    def filter(self, indexes):
        self.__decisionVariables = self.__decisionVariables[indexes]
        self.__objectives = self.__objectives[indexes]
        self.evaluated = self.evaluated[indexes]

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

    problem.evaluate(population)

    return population
