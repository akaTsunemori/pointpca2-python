import pandas as pd
from argparse import ArgumentParser
from lazypredict.Supervised import LazyRegressor
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupKFold
from os.path import exists
from os import mkdir
# from sklearn.model_selection import train_test_split


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


def group_k_fold_regression(df_MATLAB: pd.DataFrame, df_Python: pd.DataFrame):
    X_MATLAB = df_MATLAB[FEATURES]
    y_MATLAB = df_MATLAB['SCORE']
    X_Python = df_Python[FEATURES]
    y_Python = df_Python['SCORE']
    groups = df_MATLAB['REF']
    gkf = GroupKFold(n_splits=len(groups.unique())) # Adjust n_splits accordingly
    dataframes_MATLAB = list()
    dataframes_Python = list()
    fold = 0
    for train_idx, test_idx in gkf.split(X_MATLAB, y_MATLAB, groups):
        X_train, X_test = X_MATLAB.iloc[train_idx], X_MATLAB.iloc[test_idx]
        y_train, y_test = y_MATLAB.iloc[train_idx], y_MATLAB.iloc[test_idx]
        testing_groups = ', '.join(i for i in groups.iloc[test_idx].unique())
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_MATLAB.append(models)
        X_train, X_test = X_Python.iloc[train_idx], X_Python.iloc[test_idx]
        y_train, y_test = y_Python.iloc[train_idx], y_Python.iloc[test_idx]
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_Python.append(models)
        fold += 1
    result_df_MATLAB = pd.concat(dataframes_MATLAB)
    result_df_MATLAB = result_df_MATLAB.reset_index(drop=False)
    result_df_Python = pd.concat(dataframes_Python)
    result_df_Python = result_df_Python.reset_index(drop=False)
    cols_to_move = ['Testing Groups', 'Fold']
    for col in cols_to_move:
        column_to_move = result_df_MATLAB.pop(col)
        result_df_MATLAB.insert(0, col, column_to_move)
        column_to_move = result_df_Python.pop(col)
        result_df_Python.insert(0, col, column_to_move)
    return result_df_MATLAB, result_df_Python


def get_args():
    parser = ArgumentParser()
    parser.add_argument('csv_path_MATLAB', type=str,
                        help='./results/apsipa_pointpca2_MATLAB_cleaned.csv')
    parser.add_argument('csv_path_Python', type=str,
                        help='./results/apsipa_pointpca2_Python_cleaned.csv')
    parser.add_argument('dataset_name', type=str,
                        help='Example: APSIPA')
    return parser.parse_args()


def main(args):
    csv_path_Python = args.csv_path_Python
    csv_path_MATLAB = args.csv_path_MATLAB
    dataset_name = args.dataset_name
    df_Python = pd.read_csv(csv_path_Python, index_col=0)
    df_MATLAB = pd.read_csv(csv_path_MATLAB, index_col=0)
    # When using train_test_split (function args need to be adapted)
    # MATLAB = compute_regression(df_Python[FEATURES].values, df_Python['SCORE'].values, 'regression_MATLAB.csv')
    # Python = compute_regression(df_MATLAB[FEATURES].values, df_MATLAB['SCORE'].values, 'regression_Python.csv')
    # MATLAB.to_csv('regression_MATLAB.csv')
    # Python.to_csv('regression_Python.csv')
    MATLAB = one_group_out_regression(df_MATLAB)
    Python = one_group_out_regression(df_Python)
    if not exists('./regressions'):
        mkdir('regressions')
    MATLAB.to_csv(f'regressions/{dataset_name}_MATLAB_regression_LeaveOneGroupOut.csv')
    Python.to_csv(f'regressions/{dataset_name}_Python_regression_LeaveOneGroupOut.csv')
    MATLAB, Python = group_k_fold_regression(df_MATLAB, df_Python)
    MATLAB.to_csv(f'regressions/{dataset_name}_MATLAB_regression_GroupKFold.csv')
    Python.to_csv(f'regressions/{dataset_name}_Python_regression_GroupKFold.csv')


if __name__ == '__main__':
    main(get_args())