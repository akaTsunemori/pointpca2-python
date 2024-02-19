import pandas as pd
from scipy import stats


matlab = pd.read_csv('regression_true_LeaveOneGroupOut.csv', index_col=0).dropna()
python = pd.read_csv('regression_pred_LeaveOneGroupOut.csv', index_col=0).dropna()
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
ttests_df.to_csv('ttests_LeaveOneGroupOut.csv')
