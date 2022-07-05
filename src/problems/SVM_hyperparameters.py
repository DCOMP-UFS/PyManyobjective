from src.problems.Problem import Problem
from src.Solution import Solution

from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score

import numpy as np

class SVM_hyperparameters(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, classification=True):
        lowerBounds = [0.0 for _ in range(3)]
        upperBounds = [1.0 for _ in range(3)]

        # C
        lowerBounds[0] = 0.1
        upperBounds[0] = 1

        # kernel
        lowerBounds[1] = 0.1
        upperBounds[1] = 0.3

        # epsilon
        lowerBounds[2] = 0
        upperBounds[2] = 1

        n_objs = 3
        if classification:
            lowerBounds[1] = 0
            lowerBounds = lowerBounds[0:2]
            upperBounds = upperBounds[0:2]
            n_objs = 2

        super(SVM_hyperparameters, self).__init__(1,
                                n_objs,
                                (lowerBounds, upperBounds))
        self.problem = "SVM_hyperparameters"

        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classification = classification

    def get_model(self):
        return SVC(kernel='rbf')

    def evaluate(self, population):
        fake_params = population.getNotEvaluatedVars()

        if self.classification:
            resize_consts = np.array([50, 10])
        else:
            resize_consts = np.array([50, 10, 1])

        params = fake_params * resize_consts
        # round second param
        params[:,1] = np.round(params[:,1])

        all_means = np.zeros((params.shape[0], 1))
        for i in range(params.shape[0]):
            C = params[i][0]
            kernel = self.kernels[int(params[i][1])]
            if self.classification:
                model = SVC(C=C, kernel=kernel)
                model.fit(self.X_train, self.y_train)
                scores = cross_val_score(model, self.X_test, self.y_test, cv=3, scoring='accuracy')
                all_means[i][0] = 1.0 - scores.mean()
            else:
                epsilon = params[i][2]

                model = SVR(C=C, kernel=kernel, epsilon=epsilon)
                model.fit(self.X_train, self.y_train)
                scores = cross_val_score(model, self.X_test, self.y_test, cv=3, scoring='neg_mean_squared_error')
                all_means[i][0] = -scores.mean()
        population.setNotEvaluatedObjectives(all_means)

