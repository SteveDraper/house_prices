
import pandas as pd

from math import sqrt, log

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


def cat_to_mean_price(df, field):
    lookup = df.groupby(field)['SalePrice'].mean()
    return df[field].apply(lambda v: lookup[v])


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
        #'Street',
        #'LotShape',
        #'LandContour',
        #'Utilities',
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
        #'FireplaceQu',
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
        'GarageFinish': 'RFn',
        'FireplaceQu': 'None',
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
        'Neighborhood',
        'Utilities'
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
    for column in simple_one_hot_columns:
        modal = cleaned[column].mode()[0]
        x = cleaned[column].fillna(modal, inplace=True)
        print("Replacing missing values in column {} with {}".format(column, modal))
    for column in drop_nan_one_hot_columns:
        modal = cleaned[column].mode()[0]
        x = cleaned[column].fillna(modal, inplace=True)
        print("Replacing missing values in column {} with {}".format(column, modal))
    # cleaned['SeasonSold'] = cleaned['MoSold'].apply(lambda m: int(m/4))
    cleaned['Functional'] = cleaned['Functional'].apply(lambda c: ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'].index(c))
    cleaned['LotShape'] = cleaned['LotShape'].apply(lambda c: ['IR1', 'Reg', 'IR2', 'IR3'].index(c))
    cleaned['LandContour'] = cleaned['LandContour'].apply(lambda c: ['Lvl', 'HLS', 'Bnk', 'Low'].index(c))
    cleaned['LandSlope'] = cleaned['LandSlope'].apply(lambda c: ['Gtl', 'Mod', 'Sev'].index(c))
    cleaned['GarageFinish'] = cleaned['GarageFinish'].apply(lambda c: ['Unf', 'RFn', 'Fin'].index(c))
    #cleaned['ExterQual'] = cat_to_mean_price(cleaned, 'ExterQual')
    cleaned['ExterQual'] = cleaned['ExterQual'].apply(lambda c: qual_values.index(c))
    cleaned['BsmtQual'] = cleaned['BsmtQual'].apply(lambda c: qual_values.index(c))
    cleaned['HeatingQC'] = cleaned['HeatingQC'].apply(lambda c: qual_values.index(c))
    cleaned['KitchenQual'] = cleaned['KitchenQual'].apply(lambda c: qual_values.index(c))
    cleaned['FireplaceQu'] = cat_to_mean_price(cleaned, 'FireplaceQu')
    cleaned['GarageQual'] = cleaned['GarageQual'].apply(lambda c: qual_values.index(c))
    #cleaned['FireplaceQu'] = cleaned['FireplaceQu'].apply(lambda c: qual_values.index(c))
    cleaned['FireplaceQu'] = cat_to_mean_price(cleaned, 'FireplaceQu')
    cleaned['HasGarage'] = cleaned['GarageType'].apply(lambda a: float(not pd.isnull(a)))
    cleaned['HasBasement'] = cleaned['TotalBsmtSF'].apply(lambda a: (a > 0))
    cleaned['HasDecking'] = cleaned['WoodDeckSF'].apply(lambda a: (a > 0))
    cleaned['Street'] = cleaned['Street'].apply(lambda s: int(s == 'Pave'))
    cleaned["LotFrontage"] = cleaned.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

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
    # cleaned.loc[cleaned['LotFrontage'].isnull(), 'LotFrontage'] = 0
    # cleaned.loc[cleaned['LotFrontage'] > 0, 'LotDepth'] = cleaned['LotArea']/cleaned['LotFrontage']
    # cleaned.loc[cleaned['LotFrontage'] == 0, 'LotDepth'] = cleaned['LotArea'].apply(lambda a: sqrt(a))
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
            cleaned['Quad_' + str(column)] = cleaned[column].apply(lambda v: v*v)

    print("Cleaned data shape: ", cleaned.shape)
    return cleaned
