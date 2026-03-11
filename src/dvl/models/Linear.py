import numpy as np
from src.dvl.Model import Model
from sklearn.linear_model import LinearRegression

class LinearModel(Model):
    def __init__(self, ):
       self.regressor = LinearRegression() 

    def train(self, population, objectives):
        self.regressor.fit(objectives, population)
        return self.regressor

    def predict(self, reference_point):
        reference_point = np.asarray(reference_point)
        if reference_point.ndim == 1:
            reference_point = reference_point.reshape(1, -1)
        return self.regressor.predict(reference_point)