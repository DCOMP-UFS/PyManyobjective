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
        """@obs
            Expects a 2D array.
        """
        return self.regressor.predict(np.asarray(reference_point))
