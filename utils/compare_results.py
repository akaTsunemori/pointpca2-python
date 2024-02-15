import pandas as pd
import numpy as np


def mse(row1, row2):
    return np.mean((row1 - row2) ** 2)


FEATURES = [f'FEATURE_{i+1}' for i in range(40)]
df1 = pd.read_csv('apsipa_pointpca2_true_cleaned.csv', index_col=0)
df2 = pd.read_csv('apsipa_pointpca2_pred_cleaned.csv', index_col=0)
df1, df2 = df1[FEATURES], df2[FEATURES]

mse_values = []
for i in range(len(df1)):
    mse_values.append(mse(df1.iloc[i], df2.iloc[i]))
mse_array = np.array(mse_values)

print(f'STD:  {mse_array.std():.4f}')
print(f'MEAN: {mse_array.mean():.4f}')
