from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from helpers import model_accuracy, MidpointNormalize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def train_svr(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = RobustScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(X[selected_features])
    svr = svm.SVR(kernel='linear', C=2) ##, max_iter=200000)
    #
    # epsilon_range = [0.1]
    # C_range = [1.5,2.0,2.5]
    # param_grid = { "epsilon": epsilon_range, "C": C_range}
    #
    # gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=3), n_jobs=-1)
    #
    # gs = gs.fit(transformed, Y.reshape(-1))
    #
    # print(gs.cv_results_)
    # print(gs.best_params_)
    # print(gs.best_score_)
    #
    # scores = gs.cv_results_['mean_test_score'].reshape(len(C_range),
    #                                                    len(epsilon_range))
    #
    # stds = gs.cv_results_['std_test_score'].reshape(len(C_range),
    #                                                 len(epsilon_range))
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,2,1)
    # plt.imshow(-scores, interpolation='nearest', cmap=plt.cm.hot,
    #            norm=MidpointNormalize(vmin=-0.1, midpoint=-0.04))
    # plt.xlabel('kernel')
    # plt.ylabel('C')
    # plt.colorbar()
    # plt.xticks(np.arange(len(epsilon_range)), epsilon_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    # ax1.set_title('Validation accuracy')
    #
    # ax2 = fig.add_subplot(1,2,2)
    # plt.imshow(-stds, interpolation='nearest', cmap=plt.cm.hot,
    #            norm=MidpointNormalize(vmin=-0.1, midpoint=-0.003))
    # plt.colorbar()
    # plt.xticks(np.arange(len(epsilon_range)), epsilon_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    # ax2.set_title('Validation std-dev')
    #
    # plt.show()
    #
    # params = gs.best_params_
    # params['C'] = 2.0
    # svr = svm.SVR(**params)

    scores = cross_val_score(svr, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = svr.fit(scaler.transform(X[selected_features]), Y)
    print("Loss on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler
