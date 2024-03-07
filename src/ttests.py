import pandas as pd
from scipy import stats
from os.path import exists
from os import mkdir
from argparse import ArgumentParser


def ttests(csv_path_regression_reference,
           csv_path_regression_test,
           dataset_name,
           regression_type):
    reference = pd.read_csv(csv_path_regression_reference, index_col=0).dropna()
    test = pd.read_csv(csv_path_regression_test, index_col=0).dropna()
    ttests = {
        'Model': [],
        'p-value (Pearson)': [],
        'p_value ≤ 0.05 (Pearson)': [],
        'p-value (Spearman)': [],
        'p_value ≤ 0.05 (Spearman)': [],
    }
    regressors = reference['Model'].unique()
    alpha = 0.05
    for regressor in regressors:
        ttests['Model'].append(regressor)
        test_regressor = test[test['Model'] == regressor]
        reference_regressor = reference[reference['Model'] == regressor]
        t_stat, p_value = stats.ttest_ind(
            reference_regressor['Pearson'].to_numpy(),
            test_regressor['Pearson'].to_numpy())
        ttests['p-value (Pearson)'].append(p_value)
        ttests['p_value ≤ 0.05 (Pearson)'].append(p_value <= alpha)
        t_stat, p_value = stats.ttest_ind(
            reference_regressor['Spearman'].to_numpy(),
            test_regressor['Spearman'].to_numpy())
        ttests['p-value (Spearman)'].append(p_value)
        ttests['p_value ≤ 0.05 (Spearman)'].append(p_value <= alpha)
    ttests_df = pd.DataFrame(ttests)
    if not exists('./ttests'):
        mkdir('./ttests')
    ttests_df.to_csv(f'./ttests/{dataset_name}_{regression_type}_ttests.csv')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('csv_path_regression_reference_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_reference_GroupKFold', type=str)
    parser.add_argument('csv_path_regression_test_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_test_GroupKFold', type=str)
    parser.add_argument('dataset_name', type=str)
    return parser.parse_args()


def main(args):
    ttests(
        args.csv_path_regression_reference_LeaveOneGroupOut,
        args.csv_path_regression_test_LeaveOneGroupOut,
        args.dataset_name,
        'LeaveOneGroupOut')
    ttests(
        args.csv_path_regression_reference_GroupKFold,
        args.csv_path_regression_test_GroupKFold,
        args.dataset_name,
        'GroupKFold')


if __name__ == '__main__':
    main(get_args())