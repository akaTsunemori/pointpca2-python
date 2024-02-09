import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from scipy.stats import spearmanr, pearsonr


def custom_metric(X, y):
    return spearmanr(X, y)[0], pearsonr(X, y)[0]


def compute_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=custom_metric, random_state=0)
    models, predictions = regressor.fit(X_train, X_test, y_train, y_test)
    models = models.sort_values('Model')
    models[['Spearman', 'Pearson']]  = models['custom_metric'].apply(pd.Series)
    models = models.drop('custom_metric', axis=1)
    predictions.head()
    return models


FEATURES = [f'FEATURE_{i+1}' for i in range(40)]
df_pred = pd.read_csv('apsipa_pointpca2_pred_cleaned_v2.csv', index_col=0)
df_true = pd.read_csv('apsipa_pointpca2_true_cleaned_v2.csv', index_col=0)
true = compute_regression(df_pred[FEATURES].values, df_pred['SCORE'].values, 'regression_true.csv')
pred = compute_regression(df_true[FEATURES].values, df_true['SCORE'].values, 'regression_pred.csv')
true.to_csv('regression_true.csv')
pred.to_csv('regression_pred.csv')
