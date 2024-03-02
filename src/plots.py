import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from os.path import exists
from os import mkdir
from shutil import move


def plot(csv_path_regression_reference_LeaveOneGroupOut,
         csv_path_regression_reference_GroupKFold,
         csv_path_regression_test_LeaveOneGroupOut,
         csv_path_regression_test_GroupKFold,
         dataset_name):
    '''Plotting utility.'''
    '''

        Dataset initialization.
        Rows with NaN values will be dropped.

    '''
    df_reference_LeaveOneGroupOut = pd.read_csv(
        csv_path_regression_reference_LeaveOneGroupOut, index_col=0)
    df_reference_GroupKFold = pd.read_csv(
        csv_path_regression_reference_GroupKFold, index_col=0)
    df_test_LeaveOneGroupOut = pd.read_csv(
        csv_path_regression_test_LeaveOneGroupOut, index_col=0)
    df_test_GroupKFold = pd.read_csv(
        csv_path_regression_test_GroupKFold, index_col=0)
    df_reference_LeaveOneGroupOut = df_reference_LeaveOneGroupOut.dropna()
    df_test_LeaveOneGroupOut = df_test_LeaveOneGroupOut.dropna()
    df_reference_GroupKFold = df_reference_GroupKFold.dropna()
    df_test_GroupKFold = df_test_GroupKFold.dropna()
    # Remove Lars regressor
    df_reference_LeaveOneGroupOut = df_reference_LeaveOneGroupOut[~(df_reference_LeaveOneGroupOut == 'Lars').any(axis=1)]
    df_test_LeaveOneGroupOut = df_test_LeaveOneGroupOut[~(df_test_LeaveOneGroupOut == 'Lars').any(axis=1)]
    df_reference_GroupKFold = df_reference_GroupKFold[~(df_reference_GroupKFold == 'Lars').any(axis=1)]
    df_test_GroupKFold = df_test_GroupKFold[~(df_test_GroupKFold == 'Lars').any(axis=1)]
    # Make plots folder
    if not exists('./plots'):
        mkdir('./plots')
    '''

        Pearson and Spearman coefficients for LeaveOneGroup Out and GroupKFold.
        Both True (reference) and Pred (test) are on the same figure.

    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
    df_reference = df_reference_GroupKFold
    df_test = df_test_GroupKFold
    df_reference['Dataset'] = 'Reference'
    df_test['Dataset'] = 'Test'
    df_combined = pd.concat([df_reference, df_test], ignore_index=True)
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
    df_reference = df_reference_LeaveOneGroupOut
    df_test = df_test_LeaveOneGroupOut
    df_reference['Dataset'] = 'Reference'
    df_test['Dataset'] = 'Test'
    df_combined = pd.concat([df_reference, df_test], ignore_index=True)
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
    parser.add_argument('csv_path_regression_reference_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_reference_GroupKFold', type=str)
    parser.add_argument('csv_path_regression_test_LeaveOneGroupOut', type=str)
    parser.add_argument('csv_path_regression_test_GroupKFold', type=str)
    parser.add_argument('dataset_name', type=str)
    return parser.parse_args()


def main(args):
    plot(
        args.csv_path_regression_reference_LeaveOneGroupOut,
        args.csv_path_regression_reference_GroupKFold,
        args.csv_path_regression_test_LeaveOneGroupOut,
        args.csv_path_regression_test_GroupKFold,
        args.dataset_name)


if __name__ == '__main__':
    main(get_args())