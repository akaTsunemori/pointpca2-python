import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from scipy.stats import spearmanr, pearsonr


FEATURES = [f'FEATURE_{i+1}' for i in range(40)]


def custom_metric(X, y):
    return spearmanr(X, y)[0], pearsonr(X, y)[0]


def compute_regression(X_train, X_test, y_train, y_test):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    regressor = LazyRegressor(
        verbose=0, 
        ignore_warnings=True, 
        custom_metric=custom_metric, 
        random_state=0)
    models, predictions = regressor.fit(X_train, X_test, y_train, y_test)
    models = models.sort_values('Model')
    models[['Spearman', 'Pearson']]  = models['custom_metric'].apply(pd.Series)
    models = models.drop('custom_metric', axis=1)
    return models


def one_group_out_regression(df: pd.DataFrame):
    groups = df['REF'].unique()
    dataframes = list()
    for test in groups:
        search_group = df['REF'].str.contains(test)
        group_left_one_out = df[~search_group]
        group_test = df[search_group]
        new_dataframe = compute_regression(
                X_train=group_left_one_out[FEATURES],
                X_test=group_test[FEATURES],
                y_train=group_left_one_out['SCORE'],
                y_test=group_test['SCORE'])
        new_dataframe['REF'] = test
        dataframes.append(new_dataframe)
    result_df = pd.concat(dataframes)
    result_df = result_df.reset_index(drop=False)
    column_to_move = result_df.pop('REF')
    result_df.insert(0, 'REF', column_to_move)
    return result_df


df_pred = pd.read_csv('apsipa_pointpca2_pred_cleaned_v2.csv', index_col=0)
df_true = pd.read_csv('apsipa_pointpca2_true_cleaned_v2.csv', index_col=0)
# true = compute_regression(df_pred[FEATURES].values, df_pred['SCORE'].values, 'regression_true.csv')
# pred = compute_regression(df_true[FEATURES].values, df_true['SCORE'].values, 'regression_pred.csv')
# true.to_csv('regression_true.csv')
# pred.to_csv('regression_pred.csv')
true = one_group_out_regression(df_true)
pred = one_group_out_regression(df_pred)
true.to_csv('regression_true_LeaveOneGroupOut.csv')
pred.to_csv('regression_pred_LeaveOneGroupOut.csv')
