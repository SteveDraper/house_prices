from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from helpers import model_accuracy, MidpointNormalize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def train_lasso(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(X[selected_features])
    clf = linear_model.Lasso(alpha = 0.1, fit_intercept=True)

    estimators_range = [0.75,0.1,0.5,1]
    max_features_range = [False,True]
    param_grid = { "alpha": estimators_range, "normalize": max_features_range}

    gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=3), n_jobs=-1)

    gs = gs.fit(transformed, Y.reshape(-1))

    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

    scores = gs.cv_results_['mean_test_score'].reshape(len(estimators_range),
                                                       len(max_features_range))

    stds = gs.cv_results_['std_test_score'].reshape(len(estimators_range),
                                                    len(max_features_range))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    plt.imshow(-scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-0.1, midpoint=-0.04))
    plt.xlabel('kernel')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(max_features_range)), max_features_range, rotation=45)
    plt.yticks(np.arange(len(estimators_range)), estimators_range)
    ax1.set_title('Validation accuracy')

    ax2 = fig.add_subplot(1,2,2)
    plt.imshow(-stds, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-0.1, midpoint=-0.003))
    plt.colorbar()
    plt.xticks(np.arange(len(max_features_range)), max_features_range, rotation=45)
    plt.yticks(np.arange(len(estimators_range)), estimators_range)
    ax2.set_title('Validation std-dev')

    plt.show()

    params = gs.best_params_
    clf = linear_model.Lasso(**params)
    scores = cross_val_score(clf, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = clf.fit(scaler.transform(X[selected_features]), Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler
