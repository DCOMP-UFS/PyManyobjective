import numpy as np

class Model():
   def train(self, population, objectives):
      raise NotImplementedError

   def predict(self, reference_point: np.ndarray) -> np.ndarray:
      raise NotImplementedError