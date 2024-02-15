import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'''

    Dataset initialization.
    Rows with NaN values will be dropped.

'''
df_true_LeaveOneGroupOut = pd.read_csv(
    'regression_true_LeaveOneGroupOut.csv', index_col=0)
df_pred_LeaveOneGroupOut = pd.read_csv(
    'regression_pred_LeaveOneGroupOut.csv', index_col=0)
df_true_GroupKFold = pd.read_csv(
    'regression_true_GroupKFold.csv', index_col=0)
df_pred_GroupKFold = pd.read_csv(
    'regression_pred_GroupKFold.csv', index_col=0)
df_true_LeaveOneGroupOut = df_true_LeaveOneGroupOut.dropna()
df_pred_LeaveOneGroupOut = df_pred_LeaveOneGroupOut.dropna()
df_true_GroupKFold = df_true_GroupKFold.dropna()
df_pred_GroupKFold = df_pred_GroupKFold.dropna()

# Remove Lars regressor
df_true_LeaveOneGroupOut = df_true_LeaveOneGroupOut[~(df_true_LeaveOneGroupOut == 'Lars').any(axis=1)]
df_pred_LeaveOneGroupOut = df_pred_LeaveOneGroupOut[~(df_pred_LeaveOneGroupOut == 'Lars').any(axis=1)]
df_true_GroupKFold = df_true_GroupKFold[~(df_true_GroupKFold == 'Lars').any(axis=1)]
df_pred_GroupKFold = df_pred_GroupKFold[~(df_pred_GroupKFold == 'Lars').any(axis=1)]


'''

    4-in-1 figure containing LeaveOneGroupOut and GroupKFold,
    and the Pearson and Spearman coefficients,
    each on its own axis.

'''
# fig, ax = plt.subplots(
#     nrows=2, ncols=2, 
#     figsize=(25.6, 14.4), dpi=300)
# colors = sns.color_palette("deep")
# sns.boxplot(data=df_true_LeaveOneGroupOut, x='Pearson', y='REF', ax=ax[0, 0], color=colors[0])
# sns.boxplot(data=df_true_GroupKFold, x='Pearson', y='Model', ax=ax[0, 1], color=colors[0])
# sns.boxplot(data=df_pred_LeaveOneGroupOut, x='Pearson', y='REF', ax=ax[1, 0], color=colors[1])
# sns.boxplot(data=df_pred_GroupKFold, x='Pearson', y='Model', ax=ax[1, 1], color=colors[1])
# ax[0, 0].set_title('True Dataset - Leave One Group Out')
# ax[0, 1].set_title('True Dataset - Group K-Fold')
# ax[1, 0].set_title('Predicted Dataset - Leave One Group Out')
# ax[1, 1].set_title('Predicted Dataset - Group K-Fold')
# plt.setp(ax[:, :], xlabel='Pearson Correlation Coefficient (PCC)')
# plt.setp(ax[0, 0], ylabel='Test Group Class')
# plt.setp(ax[0, 1], ylabel='Regression Model')
# plt.setp(ax[1, 0], ylabel='Test Group Class')
# plt.setp(ax[1, 1], ylabel='Regression Model')
# plt.tight_layout()
# plt.savefig('regression_plots_pearson.png')
# fig, ax = plt.subplots(
#     nrows=2, ncols=2, 
#     figsize=(25.6, 14.4), dpi=300)
# colors = sns.color_palette("deep")
# sns.boxplot(data=df_true_LeaveOneGroupOut, x='Spearman', y='REF', ax=ax[0, 0], color=colors[0])
# sns.boxplot(data=df_true_GroupKFold, x='Spearman', y='Model', ax=ax[0, 1], color=colors[0])
# sns.boxplot(data=df_pred_LeaveOneGroupOut, x='Spearman', y='REF', ax=ax[1, 0], color=colors[1])
# sns.boxplot(data=df_pred_GroupKFold, x='Spearman', y='Model', ax=ax[1, 1], color=colors[1])
# ax[0, 0].set_title('df_true_LeaveOneGroupOut')
# ax[0, 1].set_title('df_true_GroupKFold')
# ax[1, 0].set_title('df_pred_LeaveOneGroupOut')
# ax[1, 1].set_title('df_pred_GroupKFold')
# plt.setp(ax[:, :], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
# plt.setp(ax[0, 0], ylabel='Test Group Class')
# plt.setp(ax[0, 1], ylabel='Regression Model')
# plt.setp(ax[1, 0], ylabel='Test Group Class')
# plt.setp(ax[1, 1], ylabel='Regression Model')
# plt.tight_layout()
# plt.savefig('regression_plots_spearman.png')


'''

    Pearson and Spearman coefficients for LeaveOneGroup Out and GroupKFold.
    Both True (MATLAB) and Pred (Python) are on the same figure.

'''
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
# df_true = df_true_GroupKFold
# df_pred = df_pred_GroupKFold
# df_true['Dataset'] = 'MATLAB (original)'
# df_pred['Dataset'] = 'Python (proposed)'
# df_combined = pd.concat([df_true, df_pred], ignore_index=True)
# sns.boxplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined, fliersize=0)
# ax[0].set_title('Group K-Fold')
# plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
# plt.setp(ax[0], ylabel='Regression Models')
# sns.boxplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined, fliersize=0)
# ax[1].set_title('Group K-Fold')
# plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
# plt.setp(ax[1], ylabel='Regression Models')
# plt.tight_layout()
# plt.savefig('GroupKFold_All.png')
# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
# df_true = df_true_LeaveOneGroupOut
# df_pred = df_pred_LeaveOneGroupOut
# df_true['Dataset'] = 'MATLAB (original)'
# df_pred['Dataset'] = 'Python (proposed)'
# df_combined = pd.concat([df_true, df_pred], ignore_index=True)
# sns.boxplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined, fliersize=0)
# ax[0].set_title('Leave One Group Out')
# plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
# plt.setp(ax[0], ylabel='Regression Models')
# sns.boxplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined, fliersize=0)
# ax[1].set_title('Leave One Group Out')
# plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
# plt.setp(ax[1], ylabel='Regression Models')
# plt.tight_layout()
# plt.savefig('LeaveOneGroupOut_All.png')


'''

    Violin Plot:
    Pearson and Spearman coefficients for LeaveOneGroup Out and GroupKFold.
    Both True (MATLAB) and Pred (Python) are on the same figure.

'''
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
df_true = df_true_GroupKFold
df_pred = df_pred_GroupKFold
df_true['Dataset'] = 'MATLAB (original)'
df_pred['Dataset'] = 'Python (proposed)'
df_combined = pd.concat([df_true, df_pred], ignore_index=True)
sns.violinplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined)
ax[0].set_title('Group K-Fold')
plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
plt.setp(ax[0], ylabel='Regression Models')
sns.violinplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined)
ax[1].set_title('Group K-Fold')
plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
plt.setp(ax[1], ylabel='Regression Models')
plt.tight_layout()
plt.savefig('GroupKFold_ViolinPlot.png')
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12.8, 14.4), dpi=300)
df_true = df_true_LeaveOneGroupOut
df_pred = df_pred_LeaveOneGroupOut
df_true['Dataset'] = 'MATLAB (original)'
df_pred['Dataset'] = 'Python (proposed)'
df_combined = pd.concat([df_true, df_pred], ignore_index=True)
sns.violinplot(x='Pearson', y='Model', hue='Dataset', ax=ax[0], data=df_combined)
ax[0].set_title('Leave One Group Out')
plt.setp(ax[0], xlabel='Pearson Correlation Coefficient (PCC)')
plt.setp(ax[0], ylabel='Regression Models')
sns.violinplot(x='Spearman', y='Model', hue='Dataset', ax=ax[1], data=df_combined)
ax[1].set_title('Leave One Group Out')
plt.setp(ax[1], xlabel='Spearman Ranking Order Correlation Coefficient (SROCC)')
plt.setp(ax[1], ylabel='Regression Models')
plt.tight_layout()
plt.savefig('LeaveOneGroupOut_ViolinPlot.png')
