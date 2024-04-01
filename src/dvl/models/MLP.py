from sklearn.neural_network import MLPRegressor
from src.dvl.Model import Model
import numpy as np

class MLPModel(Model):
    def __init__(self, layers:tuple):
        self.model =MLPRegressor(hidden_layer_sizes=layers)

    def train(self, population, objectives):
        self.model.fit(population, objectives)

    def predict(self, reference_point) -> np.ndarray:
        return self.model.predict(reference_point.reshape(1, -1))
