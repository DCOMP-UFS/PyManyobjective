import numpy as np
from src.dvl.Model import Model
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor


class SVRModel(Model):
    """Modelo inverso baseado em SVR, equivalente ao rótulo 'SVR' usado pelo
    Artur (models_util.getModel): make_pipeline(MultiOutputRegressor(SVR(C=0.1, gamma='auto'))).

    O SVR é univariado, por isso é envolvido em MultiOutputRegressor para
    prever todas as variáveis de decisão. Por padrão NÃO há StandardScaler
    (assim como o rótulo 'SVR' do Artur); passe standardize=True para obter o
    equivalente ao rótulo 'SVRSS'.
    """

    def __init__(self, C: float = 0.1, gamma="auto", standardize: bool = False, **kwargs):
        steps = list()
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "regressor",
            MultiOutputRegressor(SVR(C=C, gamma=gamma, **kwargs)),
        ))
        self.regressor = Pipeline(steps)

    def train(self, population, objectives):
        # Modelo inverso: aprende objetivos -> variáveis de decisão.
        self.regressor.fit(objectives, population)
        return self.regressor

    def predict(self, reference_point):
        reference_point = np.asarray(reference_point)
        if reference_point.ndim == 1:
            reference_point = reference_point.reshape(1, -1)
        return self.regressor.predict(reference_point)
