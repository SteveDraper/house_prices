import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import sqrt, log
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


def load_dataset():
    train_df = pd.read_csv('../train.csv', index_col='Id')
    test_df = pd.read_csv('../test.csv', index_col='Id')
    return pd.concat([train_df, test_df], keys=['train', 'test'])


def regress_year_to_price(data, year_field, out_field):
    mean_yearly_price = data.groupby([year_field])['SalePrice'].mean()
    reversed_mean_yearly_price = mean_yearly_price[::-1]
    reversed_smoothed_yearly_prices = reversed_mean_yearly_price.rolling(window=10, min_periods=1, center=False).mean()
    smoothed_yearly_prices = reversed_smoothed_yearly_prices[::-1]
    for year in range(2000, 2011):
        smoothed_yearly_prices[year] = mean_yearly_price[year]
    for year in range(1872, 2011):
        if not year in smoothed_yearly_prices.keys():
            smoothed_yearly_prices[year] = float('nan')
    smoothed_yearly_prices = smoothed_yearly_prices.sort_index().interpolate()
    data[out_field] = data[year_field].apply(lambda y: log(smoothed_yearly_prices[y]) - 11.7)


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
        'LotShape',
        'LandContour',
        'Utilities',
        'LotConfig',
        'LandSlope',
        'Neighborhood',
        'Condition1',
        'Condition2',
        'BldgType',
        'HouseStyle',
        'RoofStyle',
        'RoofMatl',
        'MasVnrType',
        'ExterQual',
        'ExterCond',
        'Foundation',
        'Heating',
        'HeatingQC',
        'Electrical',
        'KitchenQual',
        'Functional',
        'FireplaceQu',
        'Fence',
        'SaleType',
        'SaleCondition'
    ]
    drop_nan_one_hot_columns = [
        'MSZoning',
        'Alley',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'MiscFeature'
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
        'TotalBsmtSF': 0.
    }
    replace_missing_modal = [
        'Electrical',
        'KitchenQual',
        'Functional'
    ]
    removed_from_cleaned = [
        '1stFlrSF',
        '2ndFlrSF',
        'LotArea',
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
        'Exterior1st',
        'Exterior2nd'
    ]
    use_log_scale = [
        'TotalFloorSF'
    ]
    cleaned = raw.fillna(constant_missing_replacements)
    for column in replace_missing_modal:
        cleaned[column].fillna(cleaned[column].mode())
    cleaned['HasGarage'] = cleaned['GarageType'].apply(lambda a: float(not pd.isnull(a)))
    cleaned['HasAboveGrnd'] = (cleaned['1stFlrSF'] + cleaned['2ndFlrSF']).apply(lambda a: float(a > 0))
    cleaned['HasBasement'] = cleaned['BsmtQual'].apply(lambda a: float(not pd.isnull(a)))
    cleaned['Has2ndFloor'] = cleaned['2ndFlrSF'].apply(lambda a: float(a > 0))
    cleaned['TotalFloorSF'] = cleaned['1stFlrSF'] + cleaned['2ndFlrSF'] + cleaned['TotalBsmtSF']
    cleaned['LowQualFinSF'] /= cleaned['TotalFloorSF']
    cleaned['GrLivArea'] /= cleaned['TotalFloorSF']
    cleaned = pd.get_dummies(cleaned, columns=simple_one_hot_columns)
    cleaned = pd.get_dummies(cleaned, dummy_na=True, columns=drop_nan_one_hot_columns)
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
    cleaned['OverallQual'] = cleaned['OverallQual'].apply(lambda q: (q-1.)/9.)
    cleaned['OverallCond'] = cleaned['OverallCond'].apply(lambda q: (q-1.)/9.)
    regress_year_to_price(cleaned, 'YearBuilt', 'YearBuiltPrice')
    regress_year_to_price(cleaned, 'YearRemodAdd', 'YearRemodAddPrice')
    cleaned['MasVnrArea'] = cleaned['MasVnrArea'].apply(lambda a: log(a+1.))
    cleaned['FinishedBsmtProp'] = ((cleaned['BsmtFinSF1'] + cleaned['BsmtFinSF2']) / cleaned['TotalBsmtSF']).fillna(0.)
    cleaned['UnfinishedBsmtProp'] = (cleaned['BsmtUnfSF'] / cleaned['TotalBsmtSF']).fillna(0.)
    cleaned['TotalBsmtSF'] /= cleaned['TotalFloorSF']
    cleaned['FullBath'] += cleaned['BsmtFullBath']
    cleaned['HalfBath'] += cleaned['BsmtHalfBath']
    cleaned['GarageArea'] = cleaned['GarageArea'].apply(lambda q: q/1500)
    cleaned['HasPool'] = cleaned['PoolArea'].apply(lambda q: float(q > 0))

    for column in use_log_scale:
        cleaned[column] = cleaned[column].apply(lambda v: log(v))

    for column in removed_from_cleaned:
        del cleaned[column]

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


def train_svr(X, Y, selected_features=None, cleaned=None):
    if selected_features is None:
        selected_features = X.keys()
    if cleaned is None:
        cleaned = X
    scaler = StandardScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(X[selected_features])
    clf = svm.SVR(kernel='linear', C=1, max_iter=200000)

    kernel_range = ['linear', 'poly', 'rbf', 'sigmoid']
    C_range = [1.0,1.25,1.5,2.0,2.5,3.0]
    param_grid = { "kernel": kernel_range, "C": C_range}

    gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KFold(n_splits=3), n_jobs=-1)

    gs = gs.fit(transformed, Y.reshape(-1))

    print(gs.cv_results_)
    print(gs.best_params_)
    print(gs.best_score_)

    scores = gs.cv_results_['mean_test_score'].reshape(len(C_range),
                                                       len(kernel_range))

    stds = gs.cv_results_['std_test_score'].reshape(len(C_range),
                                                    len(kernel_range))

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    plt.imshow(-scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-0.1, midpoint=-0.04))
    plt.xlabel('kernel')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(kernel_range)), kernel_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    ax1.set_title('Validation accuracy')

    ax2 = fig.add_subplot(1,2,2)
    plt.imshow(-stds, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=-0.1, midpoint=-0.003))
    plt.colorbar()
    plt.xticks(np.arange(len(kernel_range)), kernel_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    ax2.set_title('Validation std-dev')

    plt.show()

    params = gs.best_params_
    params['C'] = 2.0
    svr = svm.SVR(**params)
    scores = cross_val_score(svr, transformed, Y, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = svr.fit(scaler.transform(X[selected_features]), Y)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, Y)))

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


def main():
    df = load_dataset()
    cleaned = clean_dataset(df)
    train = cleaned.loc['train']
    test = cleaned.loc['test']
    train_prices = train['SalePrice'].apply(lambda y: log(y)-11.5)
    del train['SalePrice']
    del test['SalePrice']
    print("{} training case, {} test, targets: {}".format(train.shape, test.shape, train_prices.shape))
    selected_features = train.keys()
    model, scaler = train_svr(train, train_prices, selected_features=selected_features)

    predictions = model.predict(scaler.transform(test[selected_features]))
    predictions = np.exp(predictions + 11.5)
    prediction_df = pd.DataFrame(predictions, columns=['SalePrice'])
    prediction_df.insert(0, 'Id', test.index)
    prediction_df.to_csv('../predictions.csv', columns=['Id', 'SalePrice'], index=False)


if __name__ == '__main__':
    main()