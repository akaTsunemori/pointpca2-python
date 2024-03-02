from os import system, mkdir
from os.path import exists
import pandas as pd


def cleanup(dataset_name, csv_path_MATLAB, csv_path_Python):
    FEATURES = [f'FEATURE_{i+1}' for i in range(40)]
    df_MATLAB = pd.read_csv(csv_path_MATLAB, index_col=0)
    df_Python = pd.read_csv(csv_path_Python, index_col=0)
    nan_indices_df1 = df_MATLAB[df_MATLAB[FEATURES].isnull().any(axis=1)].index
    nan_indices_df2 = df_Python[df_Python[FEATURES].isnull().any(axis=1)].index
    indices_to_drop = nan_indices_df1.union(nan_indices_df2)
    df_MATLAB_cleaned = df_MATLAB.drop(indices_to_drop)
    df_Python_cleaned = df_Python.drop(indices_to_drop)
    are_equal = df_MATLAB_cleaned[['SIGNAL', 'REF', 'SCORE']].equals(
        df_Python_cleaned[['SIGNAL', 'REF', 'SCORE']])
    if not are_equal:
        raise Exception(
            'Error during tables cleanup. Cleaned datasets are not equal.')
    if not exists('./results'):
        mkdir('./results')
    df_MATLAB_cleaned.to_csv(f'results/{dataset_name}_pointpca2_MATLAB_cleaned.csv')
    df_Python_cleaned.to_csv(f'results/{dataset_name}_pointpca2_Python_cleaned.csv')



batch = [2, 4, 10, 20]
MATLAB_APSIPA_PATH = 'results/APSIPA/231_APSIPA_Matlab.csv'
for factor in batch:
    cleanup(f'APSIPA-DecimateBy{factor}',
            MATLAB_APSIPA_PATH,
            f'tables/APSIPA_DecimateBy{factor}/000231_APSIPA_DecimateBy{factor}_pointpca2_Python.csv')
    system(f'python3 regressions.py \
        results/APSIPA-DecimateBy{factor}_pointpca2_MATLAB_cleaned.csv \
        results/APSIPA-DecimateBy{factor}_pointpca2_Python_cleaned.csv \
        APSIPA-DecimateBy{factor}')
    system(f'python3 plots.py \
        regressions/APSIPA-DecimateBy{factor}_MATLAB_regression_LeaveOneGroupOut.csv \
        regressions/APSIPA-DecimateBy{factor}_MATLAB_regression_GroupKFold.csv \
        regressions/APSIPA-DecimateBy{factor}_Python_regression_LeaveOneGroupOut.csv \
        regressions/APSIPA-DecimateBy{factor}_Python_regression_GroupKFold.csv \
        APSIPA-DecimateBy{factor}')
    system(f'python3 ttests.py \
        regressions/APSIPA-DecimateBy{factor}_MATLAB_regression_LeaveOneGroupOut.csv \
        regressions/APSIPA-DecimateBy{factor}_MATLAB_regression_GroupKFold.csv \
        regressions/APSIPA-DecimateBy{factor}_Python_regression_LeaveOneGroupOut.csv \
        regressions/APSIPA-DecimateBy{factor}_Python_regression_GroupKFold.csv \
        APSIPA-DecimateBy{factor}')