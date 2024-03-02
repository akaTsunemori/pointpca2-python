import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os.path import exists
from os import mkdir
from shutil import move


def plot(csv_path_regression_MATLAB_LeaveOneGroupOut,
         csv_path_regression_MATLAB_GroupKFold,
         csv_path_regression_Python_LeaveOneGroupOut,
         csv_path_regression_Python_GroupKFold,
         dataset_name):
    '''Plotting utility.'''
    '''

        Dataset initialization.
        Rows with NaN values will be dropped.

    '''
    df_MATLAB_LeaveOneGroupOut = pd.read_csv(
        csv_path_regression_MATLAB_LeaveOneGroupOut, index_col=0)
    df_MATLAB_GroupKFold = pd.read_csv(
        csv_path_regression_MATLAB_GroupKFold, index_col=0)
    df_Python_LeaveOneGroupOut = pd.read_csv(
        csv_path_regression_Python_LeaveOneGroupOut, index_col=0)
    df_Python_GroupKFold = pd.read_csv(
        csv_path_regression_Python_GroupKFold, index_col=0)
    df_MATLAB_LeaveOneGroupOut = df_MATLAB_LeaveOneGroupOut.dropna()
    df_Python_LeaveOneGroupOut = df_Python_LeaveOneGroupOut.dropna()
    df_MATLAB_GroupKFold = df_MATLAB_GroupKFold.dropna()
    df_Python_GroupKFold = df_Python_GroupKFold.dropna()
    # Remove Lars regressor
    df_MATLAB_LeaveOneGroupOut = df_MATLAB_LeaveOneGroupOut[~(df_MATLAB_LeaveOneGroupOut == 'Lars').any(axis=1)]
    df_Python_LeaveOneGroupOut = df_Python_LeaveOneGroupOut[~(df_Python_LeaveOneGroupOut == 'Lars').any(axis=1)]
    df_MATLAB_GroupKFold = df_MATLAB_GroupKFold[~(df_MATLAB_GroupKFold == 'Lars').any(axis=1)]
    df_Python_GroupKFold = df_Python_GroupKFold[~(df_Python_GroupKFold == 'Lars').any(axis=1)]
    # Make plots folder
    if not exists('./plots'):
        mkdir('./plots')
    '''

        Pearson and Spearman coefficients for LeaveOneGroup Out and GroupKFold.
        Both True (MATLAB) and Pred (Python) are on the same figure.

    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
    df_MATLAB = df_MATLAB_GroupKFold
    df_Python = df_Python_GroupKFold
    df_MATLAB['Dataset'] = 'MATLAB (original)'
    df_Python['Dataset'] = 'Python (proposed)'
    df_combined = pd.concat([df_MATLAB, df_Python], ignore_index=True)
    sns.boxplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined, fliersize=0)
    ax[0].set_title('Group K-Fold')
    plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
    plt.setp(ax[0], ylabel='Regression Models')
    sns.boxplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined, fliersize=0)
    ax[1].set_title('Group K-Fold')
    plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
    plt.setp(ax[1], ylabel='Regression Models')
    plt.tight_layout()
    filename = f'{dataset_name}_GroupKFold.png'
    plt.savefig(filename)
    move(filename, f'plots/{filename}')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
    df_MATLAB = df_MATLAB_LeaveOneGroupOut
    df_Python = df_Python_LeaveOneGroupOut
    df_MATLAB['Dataset'] = 'MATLAB (original)'
    df_Python['Dataset'] = 'Python (proposed)'
    df_combined = pd.concat([df_MATLAB, df_Python], ignore_index=True)
    sns.boxplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined, fliersize=0)
    ax[0].set_title('Leave One Group Out')
    plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
    plt.setp(ax[0], ylabel='Regression Models')
    sns.boxplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined, fliersize=0)
    ax[1].set_title('Leave One Group Out')
    plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
    plt.setp(ax[1], ylabel='Regression Models')
    plt.tight_layout()
    filename = f'{dataset_name}_LeaveOneGroupOut.png'
    plt.savefig(filename)
    move(filename, f'plots/{filename}')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('csv_path_regression_MATLAB_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_MATLAB_GroupKFold', type=str)
    parser.add_argument('csv_path_regression_Python_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_Python_GroupKFold', type=str)
    parser.add_argument('dataset_name', type=str)
    return parser.parse_args()


def main(args):
    plot(
        args.csv_path_regression_MATLAB_LeaveOneGroupOut,
        args.csv_path_regression_MATLAB_GroupKFold,
        args.csv_path_regression_Python_LeaveOneGroupOut,
        args.csv_path_regression_Python_GroupKFold,
        args.dataset_name)


if __name__ == '__main__':
    main(get_args())