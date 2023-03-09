from src.problems.Problem import Problem

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

from src.problems.scores import sensitivity_func, specificity_func

import numpy as np

class SVM_hyperparameters_sen_spe_2(Problem):
    def __init__(self, X_train, y_train):
        lowerBounds = [0.0 for _ in range(2)]
        upperBounds = [1.0 for _ in range(2)]

        # C
        lowerBounds[0] = 0
        upperBounds[0] = 1

        # gamma
        lowerBounds[1] = 0
        upperBounds[1] = 1

        n_objs = 2

        super(SVM_hyperparameters_sen_spe_2, self).__init__(2,
                                n_objs,
                                (lowerBounds, upperBounds))
        self.problem = "SVM_hyperparameters_sen_spe_2"

        self.X_train = X_train
        self.y_train = y_train

        self.lo_values = np.array([1e-5, 1e-10])
        self.hi_values = np.array([1e2, 2*1e1])
        self.resize_consts = self.hi_values - self.lo_values

    def evaluate(self, population):
        fake_params = population.getNotEvaluatedVars()

        params = fake_params * self.resize_consts + self.lo_values

        all_means = np.zeros((params.shape[0], 2))
        for i in range(params.shape[0]):
            C = params[i][0]
            gamma = params[i][1]
            model = SVC(C=C, gamma=gamma)
            sensitivity_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring=make_scorer(sensitivity_func))
            specificity_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring=make_scorer(specificity_func))
            all_means[i][0] = 1.0 - sensitivity_scores.mean()
            all_means[i][1] = 1.0 - specificity_scores.mean()
        population.setNotEvaluatedObjectives(all_means)

    def get_config_accuracy(self, config):
        params = config * self.resize_consts + self.lo_values
        C = params[0]
        gamma = params[1]
        model = SVC(C=C, gamma=gamma)
        accuracy = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring="accuracy")
        return accuracy.mean()

    def get_config_model(self, config):
        params = config * self.resize_consts + self.lo_values
        C = params[0]
        gamma = params[1]
        model = SVC(C=C, gamma=gamma)
        return model


