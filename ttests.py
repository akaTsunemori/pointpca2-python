import pandas as pd
from scipy import stats
from os.path import exists
from os import mkdir
from argparse import ArgumentParser


def ttests(csv_path_regression_MATLAB,
           csv_path_regression_Python,
           dataset_name,
           regression_type):
    matlab = pd.read_csv(csv_path_regression_MATLAB, index_col=0).dropna()
    python = pd.read_csv(csv_path_regression_Python, index_col=0).dropna()
    ttests = {
        'Model': [],
        'Pearson': [],
        'Pearson p_value <= 0.05': [],
        'Spearman': [],
        'Spearman p_value <= 0.05': [],
    }
    regressors = matlab['Model'].unique()
    alpha = 0.05
    for regressor in regressors:
        ttests['Model'].append(regressor)
        python_regressor = python[python['Model'] == regressor]
        matlab_regressor = matlab[matlab['Model'] == regressor]
        t_stat, p_value = stats.ttest_ind(
            matlab_regressor['Pearson'].to_numpy(),
            python_regressor['Pearson'].to_numpy())
        ttests['Pearson'].append(p_value)
        ttests['Pearson p_value <= 0.05'].append(p_value <= alpha)
        t_stat, p_value = stats.ttest_ind(
            matlab_regressor['Spearman'].to_numpy(),
            python_regressor['Spearman'].to_numpy())
        ttests['Spearman'].append(p_value)
        ttests['Spearman p_value <= 0.05'].append(p_value <= alpha)
    ttests_df = pd.DataFrame(ttests)
    if not exists('./ttests'):
        mkdir('./ttests')
    ttests_df.to_csv(f'./ttests/{dataset_name}_{regression_type}_ttests.csv')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('csv_path_regression_MATLAB_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_MATLAB_GroupKFold', type=str)
    parser.add_argument('csv_path_regression_Python_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_Python_GroupKFold', type=str)
    parser.add_argument('dataset_name', type=str)
    return parser.parse_args()


def main(args):
    ttests(
        args.csv_path_regression_MATLAB_LeaveOneGroupOut,
        args.csv_path_regression_Python_LeaveOneGroupOut,
        args.dataset_name,
        'LeaveOneGroupOut')
    ttests(
        args.csv_path_regression_MATLAB_GroupKFold,
        args.csv_path_regression_Python_GroupKFold,
        args.dataset_name,
        'GroupKFold')


if __name__ == '__main__':
    main(get_args())