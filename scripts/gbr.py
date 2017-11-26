from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import numpy as np
from helpers import model_accuracy, MidpointNormalize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def train_gbr(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    #scaler = StandardScaler(with_mean=True, with_std=True)
    scaler = RobustScaler()
    scaler.fit(cleaned)
    transformed = scaler.transform(X)

    params = {
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "loss": 'ls',
        "max_features": 60,
        # "min_samples_leaf": 1,
        # "min_samples_split": 2,
        "max_depth": 2
    }

    param_grid = {
        "n_estimators": [600,800,1000,1200, 1400, 1600, 1800],
        # "learning_rate": [0.1, 0.08, 0.05, 0.03, 0.02],
        # "subsample": [ 0.6, 0.65, 0.7, 0.75, 0.8 ],
        # "loss": ['ls', 'lad', 'huber', 'quantile']
        "max_features": [10,20, 30, 40, 50, 60],
        #"min_samples_leaf": [1, 2],
       #  "min_samples_split": [2, 3, 4, 5, 6],
       # "max_depth": [2, 3]
    }


    gbr = GradientBoostingRegressor(**params).fit(transformed, Y)
    importances = pd.Series(gbr.feature_importances_, index=selected_features).sort_values(ascending=False)
    print("Important features: {}", list(importances.keys()[:20]))
    # important_feats = list(importances.head(200).keys())
    # scaler = StandardScaler()
    # scaler.fit(cleaned[important_feats])
    # transformed = scaler.transform(X[important_feats])

    # best_params = grid_search_gbr(gbr, transformed, Y, param_grid)
    # params.update(best_params)
    # gbr = GradientBoostingRegressor(**params)

    scores = cross_val_score(gbr, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)
    print("mean CV score: ", scores.mean())
    print("std CV score: ", scores.std())

    model = gbr.fit(transformed, Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))


    return model, scaler #, important_feats


def grid_search_gbr(gbr, X, Y, param_grid):
    feats = list(param_grid.keys())
    k1 = feats[0]
    k2 = feats[1]
    dim1 = param_grid[k1]
    dim2 = param_grid[k2]

    gs = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=3), n_jobs=-1)

    gs = gs.fit(X, Y.values.reshape(-1))

    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

    if len(feats) > 2:
        return gs.best_params_

    scores = gs.cv_results_['mean_test_score'].reshape(len(dim2),
                                                       len(dim1))

    stds = gs.cv_results_['std_test_score'].reshape(len(dim2),
                                                    len(dim1))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    plt.imshow(-scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.013, midpoint=0.016))
    plt.xlabel(k1)
    plt.ylabel(k2)
    plt.colorbar()
    plt.xticks(np.arange(len(dim1)), dim1, rotation=45)
    plt.yticks(np.arange(len(dim2)), dim2)
    ax1.set_title('Validation accuracy')

    ax2 = fig.add_subplot(1,2,2)
    plt.imshow(-stds, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-0.003, midpoint=-0.002))
    plt.colorbar()
    plt.xticks(np.arange(len(dim1)), dim1, rotation=45)
    plt.yticks(np.arange(len(dim2)), dim2)
    ax2.set_title('Validation std-dev')

    plt.show()

    return gs.best_params_

