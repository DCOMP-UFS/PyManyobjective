from src.problems.Problem import Problem
from src.Solution import Solution

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

from sklearn.metrics import accuracy_score

class Keras_NN(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, input_dim, n_classes):
        lowerBounds = [0.0 for _ in range(9)]
        upperBounds = [1.0 for _ in range(9)]

        # layers
        lowerBounds[0] = 0.1
        upperBounds[0] = 0.3

        # neurons
        lowerBounds[1] = 0.08
        upperBounds[1] = 0.24

        # loss_rate
        lowerBounds[6] = 0.01
        upperBounds[6] = 0.5

        # epochs
        lowerBounds[7] = 0.01
        upperBounds[7] = 0.2

        # batch_size
        lowerBounds[8] = 0.01
        upperBounds[8] = 0.3

        super(Keras_NN, self).__init__(1,
                                9,
                                (lowerBounds, upperBounds))
        self.problem = "Keras_NN"

        self.losses = ['binary_crossentropy', 'hinge']
        self.optimizers = ['adam', 'sgd']
        self.activations = ['relu', 'tanh']
        self.initializers = ['random_uniform', 'normal']

        #
        #
        #

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # assume flatenned data
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.model = Sequential()

    def evaluate(self, solution):
        params = solution.decisionVariables

        hidden_layers = int(np.round(params[0] * 10))
        neurons = int(round(params[1] * 100))
        loss = self.losses[int(round(params[2]))]
        optimizer = self.optimizers[int(round(params[3]))]
        activation = self.activations[int(round(params[4]))]
        kernel_initializer = self.initializers[int(round(params[5]))]
        loss_rate = params[6]
        epochs = int(round(params[7] * 100))
        batch_size = int(round(params[8] * 100))

        print(neurons, epochs, batch_size)

        model = Sequential()

        model.add(Dense(neurons, input_dim=self.input_dim, activation=activation))
        for i in range(1, hidden_layers):
            model.add(Dense(neurons, activation=activation))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        history = model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred = model.predict(self.X_test)
        # Converting predictions to label
        pred = list()
        for i in range(len(y_pred)):
            pred.append(np.argmax(y_pred[i]))
        # Converting one hot encoded test label to label
        test = list()
        for i in range(len(self.y_test)):
            test.append(np.argmax(self.y_test[i]))

        solution.objectives[0] = 1.0 - accuracy_score(pred, test)
        return solution

