import pandas as pd
import time
import tracemalloc
from argparse import ArgumentParser
from os.path import exists
from os import mkdir
from shutil import move
from pointpca2 import pointpca2


def generate_features(dataset_name, dataset_csv, decimation_factor):
    if not exists('./tables'):
        mkdir('tables')
    df_dataset = pd.read_csv(dataset_csv)
    df_dataset['SIGNAL'] = df_dataset['SIGNAL'].str.strip()
    df_dataset['REF'] = df_dataset['REF'].str.strip()
    df_dataset['LOCATION'] = df_dataset['LOCATION'].str.strip()
    df_dataset['REFLOCATION'] = df_dataset['REFLOCATION'].str.strip()
    features_columns = [f'FEATURE_{i+1}' for i in range(40)]
    common_columns = ['SIGNAL', 'REF', 'SCORE']
    extra_columns = ['Time Taken (s)', 'Peak RAM Usage (MB)']
    checkpoint_path = f'./tables/{dataset_name}_pointpca2_checkpoint.csv'
    checkpoint_path_bak = f'./tables/{dataset_name}_pointpca2_checkpoint.bak'
    if exists(checkpoint_path):
        df_features = pd.read_csv(checkpoint_path, index_col=0)
        print(f'Loaded checkpoint at {checkpoint_path}')
    else:
        df_features = pd.DataFrame(columns=common_columns+features_columns+extra_columns)
        df_features[common_columns] = df_dataset[common_columns]
        print('No checkpoint found, starting from scratch.')
    tracemalloc.start()
    for index, row in df_dataset.iterrows():
        signal, ref = row['SIGNAL'], row['REF']
        signal_location, ref_location = row['LOCATION'], row['REFLOCATION']
        print(f'{index+1}/{len(df_dataset)}')
        print('REF/SIGNAL:', ref, signal)
        if not df_features.iloc[index].isna().any():
            print('Found checkpoint, skipping...')
            continue
        try:
            tracemalloc.clear_traces()
            time_0 = time.time()
            pointpca2_features = pointpca2(
                f'{signal_location}/{signal}',
                f'{ref_location}/{ref}',
                decimation_factor)
            time_1 = time.time()
            _, peak_memory = tracemalloc.get_traced_memory()
        except Exception as e:
            print(e)
            continue
        for i in range(len(features_columns)):
            df_features.at[index, features_columns[i]] = pointpca2_features[i]
        df_features.at[index, 'Time Taken (s)'] = time_1 - time_0
        df_features.at[index, 'Peak RAM Usage (MB)'] = peak_memory / (1024 * 1024)
        print('\tSaving checkpoint to disk')
        if exists(checkpoint_path):
            move(checkpoint_path, checkpoint_path_bak)
        df_features.to_csv(checkpoint_path)
    tracemalloc.stop()


def get_args():
    parser = ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='Example: APSIPA')
    parser.add_argument('dataset_csv', type=str,
                        help='Example: /home/user/Documents/APSIPA/apsipa.csv')
    parser.add_argument('decimation_factor', type=int,
                        help='Decimation factor. 1 means no decimation, 2 means half of the dataset will be decimated.',
                        default=None)
    return parser.parse_args()


def main(args):
    dataset_name = args.dataset_name
    dataset_csv = args.dataset_csv
    decimation_factor = args.decimation_factor
    generate_features(dataset_name, dataset_csv, decimation_factor)


if __name__ == '__main__':
    main(get_args())
