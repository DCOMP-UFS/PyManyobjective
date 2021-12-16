from src.problems.Problem import Problem
from src.Solution import Solution

from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score

import numpy as np

class SVM_hyperparameters(Problem):
    def __init__(self, X_train, y_train, X_test, y_test):
        lowerBounds = [0.0 for _ in range(2)]
        upperBounds = [1.0 for _ in range(2)]

        # C
        lowerBounds[0] = 0.0000000001
        upperBounds[0] = 1

        # kernel
        lowerBounds[1] = 0
        upperBounds[1] = 0.3

        super(SVM_hyperparameters, self).__init__(1,
                                2,
                                (lowerBounds, upperBounds))
        self.problem = "SVM_hyperparameters"

        self.kernels = ['linear','poly','rbf','sigmoid']

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_model(self):
        return SVC(kernel='rbf')

    def evaluate(self, solution):
        params = solution.decisionVariables

        C = params[0] * 50
        kernel = self.kernels[int(round(params[1] * 10))]

        model = SVC(C=C, kernel=kernel)
        model.fit(self.X_train, self.y_train)

        scores = cross_val_score(model, self.X_test, self.y_test, cv=3, scoring='accuracy')

        solution.objectives[0] = 1.0 - scores.mean()
        return solution
