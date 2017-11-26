
import pandas as pd
import numpy as np
from math import sqrt, log, isnan, exp
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from ens import train_ens
from gbr import train_gbr
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


def main():
    nans = lambda df: df[df.isnull().any(axis=1)]

    apply_residuals = False

    df = load_dataset()
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
    model, scaler = train_gbr(pca_train, train_log_prices)

    # remove outliers
    pca_train['prediction'] = model.predict(scaler.transform(pca_train))
    pca_train['error'] = pca_train['prediction'] - train_log_prices
    pca_train['SalePrice'] = train_log_prices
    # sorted_by_error = pca_train['error'].sort_values(ascending=False)
    # sorted_by_error.hist()
    # plt.show()
    train_no_outliers = pca_train.loc[abs(pca_train['error']) <= 0.2]
    print("Removed {} outliers".format(pca_train.shape[0] - train_no_outliers.shape[0]))
    train_log_prices = train_no_outliers['SalePrice']
    del train_no_outliers['error']
    del train_no_outliers['prediction']
    del train_no_outliers['SalePrice']
    model, scaler = train_gbr(train_no_outliers, train_log_prices)

    # construct residual errors
    if apply_residuals:
        train_predicted = np.exp(model.predict(scaler.transform(pca_train)) + 11.5)
        residuals = train_prices - train_predicted
        train['ResidualError'] = residuals

        rms = sqrt(mean_squared_error(train_prices, train_predicted))
        print("Training error before residual model: ", rms)

        selected_linear_features = analyse_correlations(train, 'ResidualError', False)

        train_linear = train[selected_linear_features]
        test_linear = test[selected_linear_features]

        residual_model, residual_scaler = train_gbr(train_linear, residuals)

        train_predicted_residuals = residual_model.predict(residual_scaler.transform(train_linear))
        train_predicted += train_predicted_residuals

        rms = sqrt(mean_squared_error(train_prices, train_predicted))
        print("Training error after residual model: ", rms)

    predictions = model.predict(scaler.transform(pca_test))

    model2, scaler2 = train_ens(train_no_outliers, train_log_prices)
    predictions2 = model2.predict(scaler2.transform(pca_test))

    predictions += predictions2
    predictions = np.exp(predictions/2 + 11.5)

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