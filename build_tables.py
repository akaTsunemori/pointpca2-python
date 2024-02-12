import pandas as pd
import numpy as np
import matlab.engine
from pointpca2 import lc_pointpca
from os.path import exists
from os import mkdir, listdir


def get_latest_csv(filenames):
    if not filenames:
        return None, None
    latest_true = None
    latest_pred = None
    max_true_number = -1
    max_pred_number = -1
    for filename in filenames:
        parts = filename.split('_')
        if len(parts) != 4:
            return None, None
        number, _, _, status_with_extension = parts
        status = status_with_extension.split('.')[0]
        try:
            number = int(number)
        except ValueError:
            return None, None
        if status == 'true' and number > max_true_number:
            max_true_number = number
            latest_true = filename
        elif status == 'pred' and number > max_pred_number:
            max_pred_number = number
            latest_pred = filename
    return latest_true, latest_pred


eng = matlab.engine.start_matlab()
eng.addpath('/home/arthurc/Documents/pointpca2/Matlab_FeatureExtraction/lib', nargout=0)
print('MATLAB engine for Python successfully started!')
if not exists('./tables'):
    mkdir('tables')
df_dataset = pd.read_csv('/home/arthurc/Documents/APSIPA/apsipa.csv')
df_dataset['SIGNAL'] = df_dataset['SIGNAL'].str.strip()
df_dataset['REF'] = df_dataset['REF'].str.strip()
df_dataset['LOCATION'] = df_dataset['LOCATION'].str.strip()
df_dataset['REFLOCATION'] = df_dataset['REFLOCATION'].str.strip()
df_dataset['LOCATION'] = df_dataset['LOCATION'].str.replace(
    '/home/pedro/databases/QualityDatabases/PointClouds/reference_APSIPA/', '/home/arthurc/Documents/APSIPA/')
df_dataset['REFLOCATION'] = df_dataset['REFLOCATION'].str.replace(
    '/home/pedro/databases/QualityDatabases/PointClouds/reference_APSIPA/', '/home/arthurc/Documents/APSIPA/')
FEATURES = [f'FEATURE_{i+1}' for i in range(40)]

tables = listdir('./tables')
true_filename, pred_filename = get_latest_csv(tables)
if not true_filename and not pred_filename:
    raise Exception('tables folder must be empty or contain only csv files generated by this script!')
if not pred_filename:
    df_result_pred = pd.DataFrame(columns=['SIGNAL', 'REF', 'SCORE', *FEATURES])
    df_result_pred[['SIGNAL', 'REF', 'SCORE']] = df_dataset[['SIGNAL', 'REF', 'SCORE']]
else:
    df_result_pred = pd.read_csv(f'tables/{pred_filename}', index_col=0)
if not true_filename:
    df_result_true = pd.DataFrame(columns=['SIGNAL', 'REF', 'SCORE', *FEATURES])
    df_result_true[['SIGNAL', 'REF', 'SCORE']] = df_dataset[['SIGNAL', 'REF', 'SCORE']]
else:
    df_result_true = pd.read_csv(f'tables/{true_filename}', index_col=0)

for index, row in df_dataset.iterrows():
    output_true = f'tables/{index:03}_apsipa_pointpca2_true.csv'
    output_pred = f'tables/{index:03}_apsipa_pointpca2_pred.csv'
    exists_true = exists(output_true)
    exists_pred = exists(output_pred)
    signal, ref = row['SIGNAL'], row['REF']
    signal_location, ref_location = row['LOCATION'], row['REFLOCATION']
    print(f'{index}/{len(df_dataset)}')
    print('REF/SIGNAL:', ref, signal)
    try:
        print('\tComputing lc_pointpca true')
        if not exists_true:
            lcpointpca_true = eng.lc_pointpca(
                f'{signal_location}/{signal}',
                f'{ref_location}/{ref}',
                nargout=1)
            lcpointpca_true = np.array(lcpointpca_true)
            print(f'\t\tDone!')
        else:
            print(f'\t\tFound checkpoint at "{output_true}", skipping...')
        print('\tComputing lc_pointpca predicted')
        if not exists_pred:
            lcpointpca_pred = lc_pointpca(
                f'{signal_location}/{signal}',
                f'{ref_location}/{ref}')
            print(f'\t\tDone!')
        else:
            print(f'\t\tFound checkpoint at "{output_pred}", skipping...')
    except Exception as e:
        print(e)
        continue
    for i in range(len(FEATURES)):
        if not exists_true:
            df_result_true.at[index, FEATURES[i]] = lcpointpca_true[i]
        if not exists_pred:
            df_result_pred.at[index, FEATURES[i]] = lcpointpca_pred[i]
    if not exists_true or not exists_pred:
        print('\tSaving checkpoints to disk')
    if not exists_true:
        df_result_true.to_csv(output_true)
    if not exists_pred:
        df_result_pred.to_csv(output_pred)
eng.quit()
