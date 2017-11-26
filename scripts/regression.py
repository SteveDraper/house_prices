import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import sqrt, log, isnan
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.decomposition import PCA


def load_dataset():
    train_df = pd.read_csv('../train.csv', index_col='Id')
    test_df = pd.read_csv('../test.csv', index_col='Id')
    return pd.concat([train_df, test_df], keys=['train', 'test'])


def regress_year_to_price(data, year_field, out_field):
    mean_yearly_price = data.loc[data['SalePrice'] > 0].groupby([year_field])['SalePrice'].mean()
    reversed_mean_yearly_price = mean_yearly_price[::-1]
    reversed_smoothed_yearly_prices = reversed_mean_yearly_price.rolling(window=10, min_periods=1, center=False).mean()
    smoothed_yearly_prices = reversed_smoothed_yearly_prices[::-1]
    for year in range(2000, 2011):
        smoothed_yearly_prices[year] = mean_yearly_price[year]
    for year in range(1872, 2011):
        if not year in smoothed_yearly_prices.keys():
            smoothed_yearly_prices[year] = float('nan')
    smoothed_yearly_prices = smoothed_yearly_prices.sort_index().interpolate()
    data[out_field] = data[year_field].apply(lambda y: smoothed_yearly_prices[y])


def regress_date_sold_price(data, out_field):
    mean_monthly_price = data.loc[data['SalePrice'] > 0].groupby('MoSold')['SalePrice'].mean()
    data[out_field] = data['MoSold'].apply(lambda y: mean_monthly_price[y])


def clean_dataset(raw):
    # Notes:
    #   MSZoning has some rare 'NA' values which we include in the one-hot encoding
    #   Converted LotArea into LotDepth
    #   Utilities has a few missing entries but very rare - dropping from the one-hot
    #   Normalize 'OverallQual' and 'OverallCond' into [0,1]
    #   YearBuilt and YearRemodAdd replaced by regressions to log(predicted-price)
    #   Combine Exterior1st and Exterior2nd into one one-hot vector
    #   MasVnrType fill in missing with 0 and normalize on log scale (log(value+1))
    #   One-hot without NaN all the basement qualifiers + add has_basement feature
    #   Normalized basement areas to roughly unit max
    #   Replaced basement finished/unfinished areas with proportions of total
    #   Convert floor surface areas into a total SF + boolean HasSecondFloor
    #   Make LowQualFinSF and GrLivArea also proportions of total area
    #   Combined basement and above grade baths and half baths into whole-property numbers
    #   Add HasGarage feature
    #   Dropping GarageYrBlt
    #   Pools are very rare so drop all the features apart from existence
    #   Dropping YrSold

    # Possible improvements:
    #   HouseStyle may give rise to useful inferred features
    #   Should we do something to combine basements?
    #   Normalize for variance as well as mean?
    #   Handle normalization of cleaned numeric values more generically
    simple_one_hot_columns = [
        'MSSubClass',
        'Street',
        #'LotShape',
        #'LandContour',
        'Utilities',
        'LotConfig',
        #'LandSlope',
        #'Neighborhood',
        'Condition1',
        'Condition2',
        'BldgType',
        'HouseStyle',
        'RoofStyle',
        'RoofMatl',
        'MasVnrType',
        #'ExterQual',
        'ExterCond',
        'Foundation',
        'Heating',
        #'HeatingQC',
        'Electrical',
        #'KitchenQual',
        'Functional',
        'FireplaceQu',
        'Fence',
        'SaleType',
        'SaleCondition'
    ]
    drop_nan_one_hot_columns = [
        'MSZoning',
        'Alley',
        #'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'GarageType',
        #'GarageFinish',
        #'GarageQual',
        'GarageCond',
        'MiscFeature',
        #'SeasonSold',
        #'YrSold',
        #'MoSold',
    ]
    yes_no_columns = [
        'CentralAir',
        'PavedDrive'
    ]
    constant_missing_replacements = {
        'MasVnrType': 'None',
        'MasVnrArea': 0.,
        'GarageCars': 0.,
        'GarageArea': 0.,
        'FullBath': 0.,
        'BsmtFullBath': 0.,
        'HalfBath': 0.,
        'BsmtHalfBath': 0.,
        '1stFlrSF': 0.,
        '2ndFlrSF': 0.,
        'TotalBsmtSF': 0.,
        'BsmtQual': 'TA',
        'GarageQual': 'TA',
        'GarageFinish': 'RFn'
    }
    replace_missing_modal = [
        'Electrical',
        'KitchenQual',
        'Functional'
    ]
    removed_from_cleaned = [
        '1stFlrSF',
        '2ndFlrSF',
        #'LotArea',
        'YearBuilt',
        'YearRemodAdd',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'BsmtFullBath',
        'BsmtHalfBath',
        'GarageYrBlt',
        'PoolArea',
        'PoolQC',
        'YrSold',
        'MoSold',
        'Exterior1st',
        'Exterior2nd',
        'PricePerSF',
        'Neighborhood'
    ]
    use_log_scale = [
        'TotalFloorSF',
        'FullBath',
        'HalfBath',
        'KitchenAbvGr',
        'BedroomAbvGr',
        'NeighborhoodPriceEstimate',
        'GarageArea',
        'LotArea',
        # 'OverallQual',
        # 'OverallCond',
        'MasVnrArea',
        'YearBuiltPrice',
        'YearRemodAddPrice',
        'DateSoldPrice',
        'LotFrontage',
        'TotRmsAbvGrd'
    ]
    qual_values = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    cleaned = raw.fillna(constant_missing_replacements)
    for column in replace_missing_modal:
        modal = cleaned[column].mode()[0]
        x = cleaned[column].fillna(modal, inplace=True)
        print("Replacing missing values in column {} with {}".format(column, modal))
    # cleaned['SeasonSold'] = cleaned['MoSold'].apply(lambda m: int(m/4))
    cleaned['LotShape'] = cleaned['LotShape'].apply(lambda c: ['Reg', 'IR1', 'IR2', 'IR3'].index(c))
    cleaned['LandContour'] = cleaned['LandContour'].apply(lambda c: ['Lvl', 'HLS', 'Bnk', 'Low'].index(c))
    cleaned['LandSlope'] = cleaned['LandSlope'].apply(lambda c: ['Gtl', 'Mod', 'Sev'].index(c))
    cleaned['GarageFinish'] = cleaned['GarageFinish'].apply(lambda c: ['Unf', 'RFn', 'Fin'].index(c))
    cleaned['ExterQual'] = cleaned['ExterQual'].apply(lambda c: qual_values.index(c))
    cleaned['BsmtQual'] = cleaned['BsmtQual'].apply(lambda c: qual_values.index(c))
    cleaned['HeatingQC'] = cleaned['HeatingQC'].apply(lambda c: qual_values.index(c))
    cleaned['KitchenQual'] = cleaned['KitchenQual'].apply(lambda c: qual_values.index(c))
    cleaned['GarageQual'] = cleaned['GarageQual'].apply(lambda c: qual_values.index(c))
    cleaned['HasGarage'] = cleaned['GarageType'].apply(lambda a: float(not pd.isnull(a)))
    cleaned['HasBasement'] = cleaned['TotalBsmtSF'].apply(lambda a: (a > 0))

    #cleaned['Has2ndFloor'] = cleaned['2ndFlrSF'].apply(lambda a: float(a > 0))
    cleaned['HasMultipleFloors'] = cleaned.apply(lambda row: (int(row['1stFlrSF'] > 0) + int(row['2ndFlrSF'] > 0) + int(row['TotalBsmtSF'] > 0)) > 1, axis=1)
    cleaned['TotalFloorSF'] = cleaned['1stFlrSF'] + cleaned['2ndFlrSF'] + cleaned['TotalBsmtSF']
    cleaned['LowQualFinSF'] /= cleaned['TotalFloorSF']
    cleaned['PricePerSF'] = cleaned['SalePrice'] / cleaned['TotalFloorSF']
    cost_per_sf = cleaned.loc['train'].groupby(['Neighborhood'])['PricePerSF'].mean()
    # cleaned['NeighborhoodPriceEstimate'] = cleaned.apply(lambda row: cost_per_sf[row['Neighborhood']]*row['TotalFloorSF'], axis=1)
    cleaned['NeighborhoodPriceEstimate'] = cleaned['Neighborhood'].apply(lambda n: cost_per_sf[n])
    cleaned['GrLivArea'] /= cleaned['TotalFloorSF']
    cleaned = pd.get_dummies(cleaned, dummy_na=False, columns=simple_one_hot_columns)
    cleaned = pd.get_dummies(cleaned, columns=drop_nan_one_hot_columns)
    for column in yes_no_columns:
        cleaned[column] = pd.get_dummies(cleaned[column])['Y']
    exterior_vec1 = pd.get_dummies(cleaned['Exterior1st'], prefix='ext')
    exterior_vec2 = pd.get_dummies(cleaned['Exterior2nd'], prefix='ext')
    for key in exterior_vec1:
        if key in exterior_vec2:
            cleaned[key] = exterior_vec1[key] | exterior_vec2[key]
        else:
            cleaned[key] = exterior_vec1[key]
    for key in exterior_vec2:
        if not key in exterior_vec1:
            cleaned[key] = exterior_vec2[key]
    cleaned.loc[cleaned['LotFrontage'].isnull(), 'LotFrontage'] = 0
    cleaned.loc[cleaned['LotFrontage'] > 0, 'LotDepth'] = cleaned['LotArea']/cleaned['LotFrontage']
    cleaned.loc[cleaned['LotFrontage'] == 0, 'LotDepth'] = cleaned['LotArea'].apply(lambda a: sqrt(a))
    cleaned['OverallQual'] = cleaned['OverallQual'] #.apply(lambda q: (q-1.)/9.)
    cleaned['OverallCond'] = cleaned['OverallCond'] #.apply(lambda q: (q-1.)/9.)
    regress_year_to_price(cleaned, 'YearBuilt', 'YearBuiltPrice')
    regress_year_to_price(cleaned, 'YearRemodAdd', 'YearRemodAddPrice')
    regress_date_sold_price(cleaned, 'DateSoldPrice')
    cleaned['MasVnrArea'] = cleaned['MasVnrArea'] #.apply(lambda a: log(a+1.))
    cleaned['FinishedBsmtProp'] = ((cleaned['BsmtFinSF1'] + cleaned['BsmtFinSF2']) / cleaned['TotalBsmtSF']).fillna(0.)
    cleaned['UnfinishedBsmtProp'] = (cleaned['BsmtUnfSF'] / cleaned['TotalBsmtSF']).fillna(0.)
    cleaned['TotalBsmtSF'] /= cleaned['TotalFloorSF']
    cleaned['FullBath'] += cleaned['BsmtFullBath']
    cleaned['HalfBath'] += cleaned['BsmtHalfBath']
    cleaned['GarageArea'] = cleaned['GarageArea'] #.apply(lambda q: q/1500)
    cleaned['HasPool'] = cleaned['PoolArea'].apply(lambda q: float(q > 0))

    epsilon = 0.5
    # for column in use_log_scale:
    #     cleaned[column] = cleaned[column].apply(lambda v: log(v + epsilon))

    for column in removed_from_cleaned:
        del cleaned[column]

    for column in cleaned.keys():
        if not column == 'SalePrice':
            cleaned['Log_' + str(column)] = cleaned[column].apply(lambda v: log(v + epsilon))

    print("Cleaned data shape: ", cleaned.shape)
    return cleaned


def model_accuracy(model, X, Y):
    predictions = model.predict(X)
    rms = sqrt(mean_squared_error(Y, predictions))
    return rms


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


from sklearn.ensemble import GradientBoostingRegressor


def train_gbr(X, Y, cleaned=None):
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned)
    transformed = scaler.transform(X)

    param_grid = {
        "n_estimators": [500, 600, 700, 800, 900],
        #"learning_rate": [0.5, 0.1, 0.05, 0.04]
        "subsample": [ 0.8, 0.7, 0.6],
        # "loss": ['ls', 'lad', 'huber', 'quantile']
        "max_features": [80, 90, 100, 120],
         "max_depth": [2, 3]
    }

    gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, subsample=0.8,
    max_depth = 3, max_features=80, random_state = 0, loss = 'ls').fit(transformed, Y)
    # params = grid_search_gbr(gbr, transformed, Y, param_grid)
    # gbr = GradientBoostingRegressor(**params)

    scores = cross_val_score(gbr, transformed, Y, cv=KFold(n_splits=10), scoring=model_accuracy)

    print("CV scores: ", scores)
    print("mean CV score: ", scores.mean())
    print("std CV score: ", scores.std())

    model = gbr.fit(scaler.transform(X), Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler

def grid_search_gbr(gbr, X, Y, param_grid):
    feats = list(param_grid.keys())
    k1 = feats[1]
    k2 = feats[0]
    dim1 = param_grid[k1]
    dim2 = param_grid[k2]

    gs = GridSearchCV(estimator=gbr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=3), n_jobs=-1)

    gs = gs.fit(X, Y.reshape(-1))

    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

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


def train_svr(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
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


def train_rf(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(X[selected_features])
    clf = RandomForestRegressor(bootstrap=True, oob_score=True)

    estimators_range = [100,1000,5000]
    max_features_range = ['auto','sqrt','log2',100]
    param_grid = { "n_estimators": estimators_range, "max_features": max_features_range}

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
    rf = RandomForestRegressor(**params)
    scores = cross_val_score(rf, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = rf.fit(scaler.transform(X[selected_features]), Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler


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


def train_logistic(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(X[selected_features])
    clf = linear_model.LogisticRegression()

    estimators_range = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    max_features_range = [0.1,0.75,1,1.5,2]
    param_grid = { "solver": estimators_range, "C": max_features_range}

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
    clf = linear_model.LogisticRegression(**params)
    scores = cross_val_score(clf, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = clf.fit(scaler.transform(X[selected_features]), Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

    return model, scaler

def analyse_correlations(data):
    my_data = data.copy()
    my_data['LogSalePrice'] = my_data['SalePrice'].apply(lambda p: log(p))
    del my_data['SalePrice']
    corr = my_data.corr()['LogSalePrice'].apply(lambda x: abs(x))
    del corr['LogSalePrice']
    corr = corr.nlargest(1000)

    print(corr)
    selected = []
    for key in corr.index:
        if not key.startswith('Log_'):
            if corr[key] > corr['Log_'+key]:
                selected.append(key)
        else:
            other_key = key[4:]
            if other_key in corr.index and corr[key] > corr[other_key]:
                selected.append(key)
    print(selected)
    return selected


def main():
    nans = lambda df: df[df.isnull().any(axis=1)]

    df = load_dataset()
    cleaned = clean_dataset(df)
    # hasNan = nans(cleaned)
    train = cleaned.loc['train']
    test = cleaned.loc['test']

    selected_features = analyse_correlations(train)

    train_prices = train['SalePrice'].apply(lambda y: log(y)-11.5)
    del train['SalePrice']
    del test['SalePrice']
    del cleaned['SalePrice']
    print("{} training case, {} test, targets: {}".format(train.shape, test.shape, train_prices.shape))

    train = train[selected_features]
    test = test[selected_features]
    # cleaned_extra_keys = cleaned.keys().tolist()
    # for key in train.keys():
    #     cleaned_extra_keys.remove(key)
    # for extra_key in cleaned_extra_keys:
    #     print("Key {} does not occur in training data - ignored".format(extra_key))

    # pca = PCA(n_components=231)
    # pca.fit(cleaned[selected_features])
    # pca_train = pca.transform(train)
    # pca_test = pca.transform(test)
    pca_train = train
    pca_test = test
    # selected_features = train.keys()
    model, scaler = train_gbr(pca_train, train_prices)

    predictions = model.predict(scaler.transform(pca_test))

    # train_pre_2008 = train.loc[train['YrSold'] < 2008]
    # train_2008 =  train.loc[train['YrSold'] == 2008]
    # train_post_2008 = train.loc[train['YrSold'] > 2008]
    #
    # test_pre_2008 = test.loc[test['YrSold'] < 2008]
    # test_2008 = test.loc[test['YrSold'] == 2008]
    # test_post_2008 = test.loc[test['YrSold'] > 2008]
    #
    # train_prices_pre_2008 = train_pre_2008['SalePrice'].apply(lambda y: log(y)-11.5)
    # train_prices_2008 = train_2008['SalePrice'].apply(lambda y: log(y)-11.5)
    # train_prices_post_2008 = train_post_2008['SalePrice'].apply(lambda y: log(y)-11.5)
    #
    # del train_pre_2008['SalePrice']
    # del train_2008['SalePrice']
    # del train_post_2008['SalePrice']
    #
    # del test_pre_2008['SalePrice']
    # del test_2008['SalePrice']
    # del test_post_2008['SalePrice']
    #
    # print("{} pre-2008 training case, {} test, targets: {}".format(train_pre_2008.shape, test.shape, train_prices_pre_2008.shape))
    # selected_features = list(train_pre_2008.keys())
    # selected_features.remove('YrSold')
    #
    # model_pre_2008, scaler_pre_2008 = train_gbr(train_pre_2008, train_prices_pre_2008, selected_features=selected_features)
    # predictions_pre_2008 = model_pre_2008.predict(scaler_pre_2008.transform(test_pre_2008[selected_features]))
    # model_2008, scaler_2008 = train_gbr(train_2008, train_prices_2008, selected_features=selected_features)
    # predictions_2008 = model_2008.predict(scaler_2008.transform(test_2008[selected_features]))
    # model_post_2008, scaler_post_2008 = train_gbr(train_post_2008, train_prices_post_2008, selected_features=selected_features)
    # predictions_post_2008 = model_post_2008.predict(scaler_post_2008.transform(test_post_2008[selected_features]))
    #
    # # model, scaler = train_gbr(train, train_prices, selected_features=selected_features)
    # # predictions = model.predict(scaler.transform(test[selected_features]))
    #
    # predictions = np.concatenate([predictions_pre_2008, predictions_2008, predictions_post_2008])
    # test = pd.concat([test_pre_2008, test_2008, test_post_2008])
    predictions = np.exp(predictions + 11.5)
    prediction_df = pd.DataFrame(predictions, columns=['SalePrice'])
    prediction_df.insert(0, 'Id', test.index)
    prediction_df.to_csv('../predictions.csv', columns=['Id', 'SalePrice'], index=False)


if __name__ == '__main__':
    main()