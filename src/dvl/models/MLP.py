from sklearn.neural_network import MLPRegressor
from src.dvl.Model import Model
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class MLPModel(Model):
    def __init__(
        self,
        layers: tuple = (11, 11, 11),
        random_state=None,
        standardize: bool = True,
        **kwargs,
    ):
        steps = list()
        if standardize:
            steps.append(("scaler", StandardScaler()))
        kwargs.setdefault("max_iter", 1000)
        steps.append((
            "regressor",
            MLPRegressor(
                hidden_layer_sizes=layers,
                random_state=random_state,
                **kwargs,
            ),
        ))
        self.model = Pipeline(steps)

    def train(self, population, objectives):
        self.model.fit(objectives, population)
        return self.model

    def predict(self, reference_point) -> np.ndarray:
        reference_point = np.asarray(reference_point)
        if reference_point.ndim == 1:
            reference_point = reference_point.reshape(1, -1)
        return self.model.predict(reference_point)
