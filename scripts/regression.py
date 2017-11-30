
import pandas as pd
import numpy as np
from math import sqrt, log, isnan, exp
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from model import CompositeModel
from ens import EnsModel
from gbr import GbrModel
from svr import SvrModel
from sklearn.model_selection import cross_val_score, KFold
from helpers import model_accuracy
from svr import train_svr
from lasso import train_lasso
# from logistic import train_logistic
# from rf import train_rf
from clean import clean_dataset


def load_dataset():
    train_df = pd.read_csv('../train.csv', index_col='Id')
    test_df = pd.read_csv('../test.csv', index_col='Id')
    return pd.concat([train_df, test_df], keys=['train', 'test'])


def analyse_correlations(data, target_field, log_target):
    my_data = data.copy()
    if log_target:
        my_data['__target__'] = my_data[target_field].apply(lambda p: log(p))
    else:
        my_data['__target__'] = my_data[target_field]
    del my_data[target_field]
    corr = my_data.corr()['__target__'].apply(lambda x: abs(x))
    del corr['__target__']
    corr = corr.nlargest(1000)
    print(corr.head(10))

    corr_lin = corr.loc[corr.index.to_series().apply(lambda s: not s.startswith('Log_') and not s.startswith('Quad_'))]

    selected = []
    for key in corr_lin.index:
        if corr[key] >= corr['Log_'+key] and corr[key] >= corr['Quad_' + key]:
            selected.append(key)
        elif corr['Log_' + key] >= corr['Quad_' + key]:
            selected.append('Log_'+key)
        else:
            selected.append('Quad_' + key)
    print(selected)
    return selected


def remove_outliers(pca_train,train_log_prices,  model):
    pca_train['prediction'] = model.predict(pca_train)
    pca_train['error'] = pca_train['prediction'] - train_log_prices
    pca_train['SalePrice'] = train_log_prices
    # sorted_by_error = pca_train['error'].sort_values(ascending=False)
    # sorted_by_error.hist()
    # plt.show()
    result = pca_train
    #result = pca_train.drop(pca_train[(pca_train['Log_GrLivArea']>log(4000)) & (pca_train['SalePrice']<300000)].index)
    #result = pca_train.loc[abs(pca_train['error']) <= 0.23]
    del result['error']
    del result['prediction']
    print("Removed {} outliers".format(pca_train.shape[0] - result.shape[0]))
    train_log_prices = result['SalePrice']

    del result['SalePrice']
    return result, train_log_prices


def main():
    nans = lambda df: df[df.isnull().any(axis=1)]

    apply_residuals = False

    df = load_dataset()
    df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    #df = df.drop(df[(df['OverallQual']>9.9) & (df['SalePrice']<200000)].index)
    #df = df.drop(df[(df['GarageArea']>1200) & (df['SalePrice']<300000)].index)
    cleaned = clean_dataset(df)
    # hasNan = nans(cleaned)
    train = cleaned.loc['train']
    test = cleaned.loc['test']

    selected_features = analyse_correlations(train, 'SalePrice', True)

    train_prices = train['SalePrice']
    train_log_prices = train_prices.apply(lambda y: log(y)-11.5)
    del train['SalePrice']
    del test['SalePrice']
    del cleaned['SalePrice']
    print("{} training case, {} test, targets: {}".format(train.shape, test.shape, train_log_prices.shape))

    train_log = train[selected_features]
    test_log = test[selected_features]
    # cleaned_extra_keys = cleaned.keys().tolist()
    # for key in train.keys():
    #     cleaned_extra_keys.remove(key)
    # for extra_key in cleaned_extra_keys:
    #     print("Key {} does not occur in training data - ignored".format(extra_key))

    # pca = PCA(n_components=231)
    # pca.fit(cleaned[selected_features])
    # pca_train = pca.transform(train)
    # pca_test = pca.transform(test)
    pca_train = train_log #.apply(lambda r: r.apply(lambda x: x*np.random.normal(1.0, 0.02)))
    pca_test = test_log
    # selected_features = train.keys()
    model1 = GbrModel()
    model1.fit(pca_train, train_log_prices)

    # remove outliers
    train_no_outliers, train_log_prices = remove_outliers(pca_train, train_log_prices, model1)

    model1 = GbrModel()
    model2 = EnsModel()
    model3 = SvrModel()
    model = CompositeModel([model1, model2]) #, model3])

    scores = cross_val_score(model, train_no_outliers, train_log_prices, cv=KFold(n_splits=5), scoring=model_accuracy)
    print("CV scores: ", scores)
    print("mean CV score: ", scores.mean())
    print("std CV score: ", scores.std())

    model.fit(train_no_outliers, train_log_prices)

    # construct residual errors
    if apply_residuals:
        train_predicted = np.exp(model1.predict(pca_train) + 11.5)
        residuals = train_prices - train_predicted
        train['ResidualError'] = residuals

        rms = sqrt(mean_squared_error(train_prices, train_predicted))
        print("Training error before residual model: ", rms)

        selected_linear_features = analyse_correlations(train, 'ResidualError', False)

        train_linear = train[selected_linear_features]
        test_linear = test[selected_linear_features]

        residual_model = GbrModel()
        residual_model.fit(train_linear, residuals)

        train_predicted_residuals = residual_model.predict(train_linear)
        train_predicted += train_predicted_residuals

        rms = sqrt(mean_squared_error(train_prices, train_predicted))
        print("Training error after residual model: ", rms)

    predictions = model.predict(pca_test)

    predictions = np.exp(predictions + 11.5)

    if apply_residuals:
        predicted_residuals = residual_model.predict(residual_scaler.transform(test_linear))
        predictions += predicted_residuals
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

    prediction_df = pd.DataFrame(predictions, columns=['SalePrice'])
    prediction_df.insert(0, 'Id', test.index)
    prediction_df.to_csv('../predictions.csv', columns=['Id', 'SalePrice'], index=False)


if __name__ == '__main__':
    main()