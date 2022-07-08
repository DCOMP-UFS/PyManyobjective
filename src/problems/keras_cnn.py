from src.problems.Problem import Problem
from src.Solution import Solution

import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.utils import np_utils

from sklearn.metrics import accuracy_score

import numpy as np

class Keras_CNN(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, input_dim, n_classes):
        lowerBounds = [0.0 for _ in range(13)]
        upperBounds = [1.0 for _ in range(13)]

        # kernel size 1
        lowerBounds[0] = 0.1
        upperBounds[0] = 0.4

        # kernel size 2
        lowerBounds[1] = 0.1
        upperBounds[1] = 0.4

        # activation 1
        lowerBounds[2] = 0
        upperBounds[2] = 0.2

        # activation 2
        lowerBounds[3] = 0
        upperBounds[3] = 0.2

        # kind pooling 1
        lowerBounds[4] = 0
        upperBounds[4] = 0.1

        # kind pooling 2
        lowerBounds[5] = 0
        upperBounds[5] = 0.1

        # n kernels 1
        lowerBounds[6] = 0.001
        upperBounds[6] = 0.1       

        # n kernels 2
        lowerBounds[7] = 0.001
        upperBounds[7] = 0.1

        # neurons 1
        lowerBounds[8] = 0.01
        upperBounds[8] = 0.3

        # neurons 2
        lowerBounds[9] = 0.01
        upperBounds[9] = 0.3

        # activation full 1
        lowerBounds[10] = 0
        upperBounds[10] = 0.2

        # activation full 2
        lowerBounds[11] = 0
        upperBounds[11] = 0.2

        # learning rate
        lowerBounds[12] = 0.001
        upperBounds[12] = 1

        super(Keras_CNN, self).__init__(1,
                                13,
                                (lowerBounds, upperBounds))
        self.problem = "Keras_CNN"

        self.activations = ['relu', 'sigmoid', 'tanh']
        self.poolings = ['max-pooling', 'average-pooling']
        self.optimizers = ['adam', 'sgd']
        self.initializers = ['random_uniform', 'normal']

        #
        #
        #

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        print(X_train.shape)
        print(y_train.shape)

        # assume flatenned data
        self.input_dim = input_dim
        self.n_classes = n_classes

    def evaluate(self, population):
        params = population.getNotEvaluatedVars()

        all_means = np.zeros((params.shape[0], 1))
        for i in range(params.shape[0]):
            # read i params
            kernel_size_1 = int(round(params[i][0] * 10)) * 2 + 1
            kernel_size_2 = int(round(params[i][1] * 10)) * 2 + 1
            activation_1 = self.activations[int(round(params[i][2] * 10))]
            activation_2 = self.activations[int(round(params[i][3] * 10))]
            pooling_1 = self.poolings[int(round(params[i][4] * 10))]
            pooling_2 = self.poolings[int(round(params[i][5] * 10))]
            n_kernels_1 = int(params[i][6] * 1000)
            n_kernels_2 = int(params[i][7] * 1000)
            neurons_1 = int(params[i][8] * 1000)
            neurons_2 = int(params[i][9] * 1000)
            activation_full_1 = self.activations[int(round(params[i][10] * 10))]
            activation_full_2 = self.activations[int(round(params[i][11] * 10))]
            learning_rate = params[i][12]

            print(kernel_size_1, kernel_size_2, activation_1, activation_2, pooling_1, pooling_2, n_kernels_1, n_kernels_2, neurons_1, neurons_2, activation_full_1, activation_full_2, learning_rate)

            # build CNN
            model = Sequential()
            model.add(Conv2D(n_kernels_1, kernel_size=kernel_size_1, activation=activation_1, input_shape=self.input_dim))
            if pooling_1 == 'max-pooling':
                model.add(MaxPooling2D())
            else:
                model.add(AveragePooling2D())
            model.add(Conv2D(n_kernels_2, kernel_size=kernel_size_2, activation=activation_2))
            if pooling_2 == 'max-pooling':
                model.add(MaxPooling2D())
            else:
                model.add(AveragePooling2D())
            model.add(Flatten())
            model.add(Dense(neurons_1, activation=activation_full_1))
            model.add(Dense(neurons_2, activation=activation_full_2))
            model.add(Dense(self.n_classes))

            optimizer = SGD(learning_rate=learning_rate)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            history = model.fit(self.X_train, self.y_train, epochs=30, batch_size=200, verbose=1)

            y_pred = model.predict(self.X_test)
            # Converting predictions to label
            pred = list()
            for j in range(len(y_pred)):
                pred.append(np.argmax(y_pred[j]))
            # Converting one hot encoded test label to label
            test = list()
            for j in range(len(self.y_test)):
                test.append(np.argmax(self.y_test[j]))

            print(accuracy_score(pred, test))
            print(i)
            print(all_means.shape)
            all_means[i][0] = -accuracy_score(pred, test)
        
        population.setNotEvaluatedObjectives(all_means)

