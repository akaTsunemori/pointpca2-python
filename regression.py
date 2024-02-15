import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold


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


def group_k_fold_regression(df_true: pd.DataFrame, df_pred: pd.DataFrame):
    X_true = df_true[FEATURES]
    y_true = df_true['SCORE']
    X_pred = df_pred[FEATURES]
    y_pred = df_pred['SCORE']
    groups = df_true['REF']
    gkf = GroupKFold(n_splits=len(groups.unique())) # Adjust n_splits accordingly
    dataframes_true = list()
    dataframes_pred = list()
    fold = 0
    for train_idx, test_idx in gkf.split(X_true, y_true, groups):
        X_train, X_test = X_true.iloc[train_idx], X_true.iloc[test_idx]
        y_train, y_test = y_true.iloc[train_idx], y_true.iloc[test_idx]
        testing_groups = ', '.join(i for i in groups.iloc[test_idx].unique())
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_true.append(models)
        X_train, X_test = X_pred.iloc[train_idx], X_pred.iloc[test_idx]
        y_train, y_test = y_pred.iloc[train_idx], y_pred.iloc[test_idx]
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_pred.append(models)
        fold += 1
    result_df_true = pd.concat(dataframes_true)
    result_df_true = result_df_true.reset_index(drop=False)
    result_df_pred = pd.concat(dataframes_pred)
    result_df_pred = result_df_pred.reset_index(drop=False)
    cols_to_move = ['Testing Groups', 'Fold']
    for col in cols_to_move:
        column_to_move = result_df_true.pop(col)
        result_df_true.insert(0, col, column_to_move)
        column_to_move = result_df_pred.pop(col)
        result_df_pred.insert(0, col, column_to_move)
    return result_df_true, result_df_pred

df_pred = pd.read_csv('apsipa_pointpca2_pred_cleaned.csv', index_col=0)
df_true = pd.read_csv('apsipa_pointpca2_true_cleaned.csv', index_col=0)
# When using train_test_split (function args need to be adapted)
# true = compute_regression(df_pred[FEATURES].values, df_pred['SCORE'].values, 'regression_true.csv')
# pred = compute_regression(df_true[FEATURES].values, df_true['SCORE'].values, 'regression_pred.csv')
# true.to_csv('regression_true.csv')
# pred.to_csv('regression_pred.csv')
true = one_group_out_regression(df_true)
pred = one_group_out_regression(df_pred)
true.to_csv('regression_true_LeaveOneGroupOut.csv')
pred.to_csv('regression_pred_LeaveOneGroupOut.csv')
true, pred = group_k_fold_regression(df_true, df_pred)
true.to_csv('regression_true_GroupKFold.csv')
pred.to_csv('regression_pred_GroupKFold.csv')