from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    def get_params(self, deep):
        return {}

    def set_param(self, _):
        pass

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

