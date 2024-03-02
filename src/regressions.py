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


def group_k_fold_regression(df_reference: pd.DataFrame, df_test: pd.DataFrame):
    X_reference = df_reference[FEATURES]
    y_reference = df_reference['SCORE']
    X_test = df_test[FEATURES]
    y_test = df_test['SCORE']
    groups = df_reference['REF']
    gkf = GroupKFold(n_splits=len(groups.unique())) # Adjust n_splits accordingly
    dataframes_reference = list()
    dataframes_test = list()
    fold = 0
    for train_idx, test_idx in gkf.split(X_reference, y_reference, groups):
        X_train, X_test = X_reference.iloc[train_idx], X_reference.iloc[test_idx]
        y_train, y_test = y_reference.iloc[train_idx], y_reference.iloc[test_idx]
        testing_groups = ', '.join(i for i in groups.iloc[test_idx].unique())
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_reference.append(models)
        X_train, X_test = X_test.iloc[train_idx], X_test.iloc[test_idx]
        y_train, y_test = y_test.iloc[train_idx], y_test.iloc[test_idx]
        models = compute_regression(X_train, X_test, y_train, y_test)
        models['Fold'] = fold
        models['Testing Groups'] = testing_groups
        dataframes_test.append(models)
        fold += 1
    result_df_reference = pd.concat(dataframes_reference)
    result_df_reference = result_df_reference.reset_index(drop=False)
    result_df_test = pd.concat(dataframes_test)
    result_df_test = result_df_test.reset_index(drop=False)
    cols_to_move = ['Testing Groups', 'Fold']
    for col in cols_to_move:
        column_to_move = result_df_reference.pop(col)
        result_df_reference.insert(0, col, column_to_move)
        column_to_move = result_df_test.pop(col)
        result_df_test.insert(0, col, column_to_move)
    return result_df_reference, result_df_test


def get_args():
    parser = ArgumentParser()
    parser.add_argument('csv_path_reference', type=str,
                        help='Example: ./features/apsipa_pointpca2_reference_cleaned.csv')
    parser.add_argument('csv_path_test', type=str,
                        help='Example: ./features/apsipa_pointpca2_test_cleaned.csv')
    parser.add_argument('dataset_name', type=str,
                        help='Example: APSIPA')
    return parser.parse_args()


def main(args):
    csv_path_test = args.csv_path_test
    csv_path_reference = args.csv_path_reference
    dataset_name = args.dataset_name
    df_test = pd.read_csv(csv_path_test, index_col=0)
    df_reference = pd.read_csv(csv_path_reference, index_col=0)
    # When using train_test_split (function args need to be adapted)
    # reference = compute_regression(df_test[FEATURES].values, df_test['SCORE'].values, 'regression_reference.csv')
    # test = compute_regression(df_reference[FEATURES].values, df_reference['SCORE'].values, 'regression_test.csv')
    # reference.to_csv('regression_reference.csv')
    # test.to_csv('regression_test.csv')
    reference = one_group_out_regression(df_reference)
    test = one_group_out_regression(df_test)
    if not exists('./regressions'):
        mkdir('regressions')
    reference.to_csv(f'regressions/{dataset_name}_reference_regression_LeaveOneGroupOut.csv')
    test.to_csv(f'regressions/{dataset_name}_test_regression_LeaveOneGroupOut.csv')
    reference, test = group_k_fold_regression(df_reference, df_test)
    reference.to_csv(f'regressions/{dataset_name}_reference_regression_GroupKFold.csv')
    test.to_csv(f'regressions/{dataset_name}_test_regression_GroupKFold.csv')


if __name__ == '__main__':
    main(get_args())