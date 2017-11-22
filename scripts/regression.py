import pandas as pd
import numpy as np
from math import sqrt, log
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


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
    cleaned = raw.fillna(constant_missing_replacements)
    for column in replace_missing_modal:
        cleaned[column].fillna(cleaned[column].mode())
    cleaned['HasGarage'] = cleaned['GarageType'].apply(lambda a: float(not pd.isnull(a)))
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

    for column in removed_from_cleaned:
        del cleaned[column]

    print("Cleaned data shape: ", cleaned.shape)
    return cleaned


def model_accuracy(model, X, Y):
    predictions = model.predict(X)
    rms = sqrt(mean_squared_error(Y, predictions))
    return rms


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
    scaler = StandardScaler()
    scaler.fit(cleaned[selected_features])
    transformed = scaler.transform(train[selected_features])
    clf = svm.SVR(kernel='linear', C=1, max_iter=200000)

    scores = cross_val_score(clf, transformed, train_prices, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)

    model = clf.fit(scaler.transform(train[selected_features]), train_prices)
    print("Accuracy on training set: {}".format(model_accuracy(model, transformed, train_prices)))

    predictions = model.predict(scaler.transform(test[selected_features]))
    predictions = np.exp(predictions + 11.5)
    prediction_df = pd.DataFrame(predictions, columns=['SalePrice'])
    prediction_df.insert(0, 'Id', test.index)
    prediction_df.to_csv('../predictions.csv', columns=['Id', 'SalePrice'], index=False)


if __name__ == '__main__':
    main()