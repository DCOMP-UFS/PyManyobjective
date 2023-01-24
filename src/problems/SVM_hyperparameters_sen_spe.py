class SVM_hyperparameters_sen_spe:
    def __init__(X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def evaluate(self, population):
        fake_params = population.getNotEvaluatedVars()

        resize_consts = np.array([50, 10])

        params = fake_params * resize_consts
        params[:,1] = np.round(params[:,1])

        all_means = np.zeros((params.shape[0], 2))
        for i in range(params.shape[0]):
            C = params[i][0]
            kernel = self.kernels[int(params[i][1])]
            model = SVC(C=C, kernel=kernel)
            sensitivity_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring="sensitivity") # TODO: sensitivity score
            specificity_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, score="specificity") # TODO: specificity score
            all_means[i][0] = 1.0 - sensitivity_scores.mean()
            all_means[i][1] = 1.0 - specificity_scores.mean()
        population.setNotEvaluatedObjectives(all_means)


