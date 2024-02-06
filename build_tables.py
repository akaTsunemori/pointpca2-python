import pandas as pd
import numpy as np
import matlab.engine
from pointpca2 import lc_pointpca


eng = matlab.engine.start_matlab()
eng.addpath('/home/arthurc/Documents/pointpca2/Matlab_FeatureExtraction/lib', nargout=0)
print('MATLAB engine for Python successfully started!')
FEATURES = [f'FEATURE_{i+1}' for i in range(40)]
df_result_pred = pd.DataFrame(columns=['SIGNAL', 'REF', 'SCORE', *FEATURES])
df_result_true = pd.DataFrame(columns=['SIGNAL', 'REF', 'SCORE', *FEATURES])
df_apsipa = pd.read_csv('/home/arthurc/Documents/APSIPA/apsipa.csv')
df_result_pred[['SIGNAL', 'REF', 'SCORE']] = df_apsipa[['SIGNAL', 'REF', 'SCORE']]
df_result_true[['SIGNAL', 'REF', 'SCORE']] = df_apsipa[['SIGNAL', 'REF', 'SCORE']]
df_apsipa['LOCATION'] = df_apsipa['LOCATION'].str.replace(
    '/home/pedro/databases/QualityDatabases/PointClouds/reference_APSIPA/', '/home/arthurc/Documents/APSIPA/')
df_apsipa['REFLOCATION'] = df_apsipa['REFLOCATION'].str.replace(
    '/home/pedro/databases/QualityDatabases/PointClouds/reference_APSIPA/', '/home/arthurc/Documents/APSIPA/')
for index, row in df_apsipa.iterrows():
    signal, ref = row['SIGNAL'], row['REF']
    signal, ref = signal.strip(), ref.strip()
    signal_location, ref_location = row['LOCATION'], row['REFLOCATION']
    signal_location, ref_location = signal_location.strip(), ref_location.strip()
    print(f'{index}/{df_apsipa.shape[0]}')
    print('REF/SIGNAL:', ref, signal)
    lcpointpca_pred = lc_pointpca(
        f'{signal_location}/{signal}',
        f'{ref_location}/{ref}')
    lcpointpca_true = eng.lc_pointpca(
        f'{signal_location}/{signal}',
        f'{ref_location}/{ref}',
        nargout=1)
    lcpointpca_true = np.array(lcpointpca_true)
    for i in range(len(FEATURES)):
        df_result_pred.at[index, FEATURES[i]] = lcpointpca_pred[i]
        df_result_true.at[index, FEATURES[i]] = lcpointpca_true[i]
df_result_pred.to_csv('apsipa_pointpca2_pred.csv')
df_result_true.to_csv('apsipa_pointpca2_true.csv')
eng.quit()
