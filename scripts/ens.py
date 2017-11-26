
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from helpers import model_accuracy

def train_ens(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned)
    transformed = scaler.transform(X)

    params = {
        "alpha": 0.01,
        "l1_ratio": 0.3,
       # "max_iter": 100000
    }

    param_grid = {
        "alphas": [0.003, 0.005, 0.01, 0.05, 0.1, 1],
        "l1_ratio": [.01, .1, 0.3, .5, 0.7, .9, .99],
    }

    ens = linear_model.ElasticNetCV(**param_grid, cv=KFold(n_splits=3), n_jobs=-1)
    ens = ens.fit(transformed, Y.values.reshape(-1))
    best_params = { "alpha": ens.alpha_, "l1_ratio": ens.l1_ratio_}
    print(best_params)

    params.update(best_params)
    ens = linear_model.ElasticNet(**params)

    scores = cross_val_score(ens, transformed, Y, cv=KFold(n_splits=10), scoring=model_accuracy)
    print("CV scores: ", scores)
    print("mean CV score: ", scores.mean())
    print("std CV score: ", scores.std())

    model = ens.fit(transformed, Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler #, important_feats
