import pandas as pd
from argparse import ArgumentParser
from os import system, mkdir
from os.path import exists


def cleanup(dataset_name, csv_path_reference, csv_path_test):
    FEATURES = [f'FEATURE_{i+1}' for i in range(40)]
    df_reference = pd.read_csv(csv_path_reference, index_col=0)
    df_test = pd.read_csv(csv_path_test, index_col=0)
    nan_indices_df1 = df_reference[df_reference[FEATURES].isnull().any(axis=1)].index
    nan_indices_df2 = df_test[df_test[FEATURES].isnull().any(axis=1)].index
    indices_to_drop = nan_indices_df1.union(nan_indices_df2)
    df_reference_cleaned = df_reference.drop(indices_to_drop)
    df_test_cleaned = df_test.drop(indices_to_drop)
    are_equal = df_reference_cleaned[['SIGNAL', 'REF', 'SCORE']].equals(
        df_test_cleaned[['SIGNAL', 'REF', 'SCORE']])
    if not are_equal:
        raise Exception(
            'Error during tables cleanup. Cleaned datasets are not equal.')
    if not exists('./features'):
        mkdir('./features')
    df_reference_cleaned.to_csv(f'features/{dataset_name}_pointpca2_reference_cleaned.csv')
    df_test_cleaned.to_csv(f'features/{dataset_name}_pointpca2_test_cleaned.csv')


def run_all_scripts(dataset_name, csv_path_reference, csv_path_test):
    cleanup(dataset_name, csv_path_reference, csv_path_test)
    system(f'''
        python3 regressions.py
        features/{dataset_name}_pointpca2_reference_cleaned.csv
        features/{dataset_name}_pointpca2_test_cleaned.csv
        {dataset_name}
        ''')
    system(f'''
        python3 plots.py
        regressions/{dataset_name}_reference_regression_LeaveOneGroupOut.csv
        regressions/{dataset_name}_reference_regression_GroupKFold.csv
        regressions/{dataset_name}_test_regression_LeaveOneGroupOut.csv
        regressions/{dataset_name}_test_regression_GroupKFold.csv
        {dataset_name}
        ''')
    system(f'''
        python3 ttests.py \
        regressions/{dataset_name}_reference_regression_LeaveOneGroupOut.csv \
        regressions/{dataset_name}_reference_regression_GroupKFold.csv \
        regressions/{dataset_name}_test_regression_LeaveOneGroupOut.csv \
        regressions/{dataset_name}_test_regression_GroupKFold.csv \
        {dataset_name}
        ''')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='Example: APSIPA')
    parser.add_argument('csv_path_reference', type=str,
                        help='Example: /home/user/pointca2-python/tables/APSIPA_NoDecimation_pointpca2_checkpoint.csv')
    parser.add_argument('csv_path_test', type=str,
                        help='Example: /home/user/pointca2-python/tables/APSIPA_DecimateBy2_pointpca2_checkpoint.csv')
    return parser.parse_args()


def main(args):
    dataset_name = args.dataset_name
    csv_path_reference = args.csv_path_reference
    csv_path_test = args.csv_path_test
    run_all_scripts(dataset_name, csv_path_reference, csv_path_test)


if __name__ == '__main__':
    main(get_args())