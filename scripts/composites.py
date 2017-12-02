import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from svr import SvrModel
from model import Model


class SimpleCompositeModel(Model):
    def get_params(self, deep):
        return {'models': self.models}

    def set_param(self, **params):
        self.models = params.get('models')

    def __init__(self, models=None):
        self.models = models

    def train_first_level(self, X, Y):
        pass

    def fit(self, X, Y):
        for model in self.models:
            model.fit(X, Y)

    def predict(self, X):
        acc = self.models[0].predict(X)
        for model in self.models[1:]:
            acc += model.predict(X)

        return acc/len(self.models)


class StagedCompositeModel(Model):
    def get_params(self, deep):
        return {'models': self.models, 'oof_predictions': self.oof_predictions}

    def set_param(self, **params):
        self.models = params.get('models')
        self.oof_predictions = params.get('oof_predictions')

    def __init__(self, models, oof_predictions={}):
        self.models = models
        self.oof_predictions = oof_predictions
        self.meta_model = None

    def train_first_level(self, X, Y, folds=5):
        fold_size = int((X.shape[0] + folds - 1)/folds)
        for model in self.models:
            oof = None
            for fold in range(0, folds):
                fold_start = fold*fold_size
                fold_end = (fold+1)*fold_size
                X_test = X[fold_start:fold_end]
                X_train = pd.concat([X[:fold_start], X[fold_end:]])
                Y_train = pd.concat([Y[:fold_start], Y[fold_end:]])
                model.fit(X_train, Y_train)
                Y_predict = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=[type(model).__name__])
                if oof is None:
                    oof = Y_predict
                else:
                    oof = pd.concat([oof, Y_predict])
            self.oof_predictions[type(model).__name__] = oof

    def fit(self, X, Y):
        meta_X = None
        for model in self.models:
            model_pred = self.oof_predictions[type(model).__name__].loc[X.index, :]
            if meta_X is None:
                meta_X = model_pred.copy()
            else:
                meta_X[type(model).__name__] = model_pred
            model.fit(X, Y) # ready for predict
        self.meta_model = SvrModel()
        # self.meta_model = GradientBoostingRegressor(n_estimators=500, loss='huber', learning_rate=0.05, max_depth=4, subsample=0.8)
        self.meta_model.fit(meta_X, Y)

    def predict(self, X):
        acc = pd.DataFrame(self.models[0].predict(X), columns=[type(self.models[0]).__name__])
        for model in self.models[1:]:
            acc[type(model).__name__] = model.predict(X)

        return self.meta_model.predict(acc)