import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


FEATURES = [f'FEATURE_{i+1}' for i in range(40)]


def compute_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
    print(df.head())
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(regressor.feature_importances_, index=FEATURES)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importances')


df_pred = pd.read_csv('results/v1/apsipa_pointpca2_pred_cleaned.csv', index_col=0)
df_true = pd.read_csv('results/v1/apsipa_pointpca2_true_cleaned.csv', index_col=0)
print('df_true')
compute_regression(df_true[FEATURES].values, df_true['SCORE'].values)
print('df_pred')
compute_regression(df_pred[FEATURES].values, df_pred['SCORE'].values)
plt.show()
