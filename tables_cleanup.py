import pandas as pd


FEATURES = [f'FEATURE_{i+1}' for i in range(40)]

df_true = pd.read_csv('tables/202_apsipa_pointpca2_true.csv', index_col=0)
df_pred = pd.read_csv('tables/202_apsipa_pointpca2_pred.csv', index_col=0)

nan_indices_df1 = df_true[df_true[FEATURES].isnull().any(axis=1)].index
nan_indices_df2 = df_pred[df_pred[FEATURES].isnull().any(axis=1)].index
indices_to_drop = nan_indices_df1.union(nan_indices_df2)

df_true_cleaned = df_true.drop(indices_to_drop)
df_pred_cleaned = df_pred.drop(indices_to_drop)

are_equal = df_true_cleaned[['SIGNAL', 'REF', 'SCORE']].equals(
    df_pred_cleaned[['SIGNAL', 'REF', 'SCORE']])

print(are_equal)

# df_true_cleaned.to_csv('apsipa_pointpca2_true_cleaned.csv')
# df_pred_cleaned.to_csv('apsipa_pointpca2_pred_cleaned.csv')

