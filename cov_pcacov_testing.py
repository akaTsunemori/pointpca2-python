import numpy as np
import pandas as pd
import scipy.linalg as spl
from sklearn.decomposition import PCA


# True values (from MATLAB)
COV_MATRIX_TRUE = np.asarray([
    [405.6923,  -20.9330,   16.8979],
    [-20.9330,  232.0195,    5.6073],
    [16.8979,    5.6073,   43.8997],
])
PCACOV_MATRIX_TRUE = np.asarray([
    [0.9923,    0.1143,   -0.0485],
    [-0.1160,    0.9926,   -0.0350],
    [0.0442,    0.0403,    0.9982],
])
# Test matrix
M = np.asarray([
    [6.8381,  -19.1668,   -5.0166],
    [8.6241,  -14.9650,    2.3259],
    [-0.0174,   -6.6602,   -5.6207],
    [23.6110,   -7.1837,   -0.7921],
    [22.5405,   -7.3309,    7.3537],
    [-36.1875,  -14.2786,    3.1350],
    [-19.1071,   13.6332,   -6.9694],
    [-20.2458,   14.3596,   -2.4518],
    [2.1684,   18.2854,   14.0633],
    [19.5279,   20.4373,   -4.0528],
])


# --- COV section ---
# Custom cov function
# def cov(data):
#     means = np.mean(data, axis=0)
#     centered_data = data - means
#     covariance_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
#     return covariance_matrix
# cov_matrix = cov(M)

# Numpy cov fuction
# cov_matrix = np.cov(M, rowvar=False, ddof=0)

# Pandas cov function
M_df = pd.DataFrame(M)
cov_matrix = pd.DataFrame.cov(M_df, ddof=1)
cov_matrix = cov_matrix.to_numpy()
# cov_matrix *= 1e6
# cov_matrix = cov_matrix.astype(int)
# cov_matrix = cov_matrix.astype(float)
# cov_matrix /= 1e6


# --- PCA section ---
# numpy eigh PCA
# eigenvalues, eigvecs = np.linalg.eigh(cov_matrix)
# sorted_indices = np.argsort(eigenvalues)[::-1]
# eigvecs = eigvecs[:, sorted_indices]

# sklearn PCA
# pca = PCA(n_components=3)
# pca.fit(M_df)
# eigvecs = pca.components_

# numpy eig PCA
# eigenvalues, eigvecs = np.linalg.eig(cov_matrix)
# sorted_indices = np.argsort(eigenvalues)[::-1]
# eigvecs = eigvecs[:, sorted_indices]

# scipy PCA
# eigenvalues, eigvecs = spl.eig(cov_matrix)
# sorted_indices = np.argsort(eigenvalues)[::-1]
# eigvecs = eigvecs[:, sorted_indices]

# scipy PCA 2
# eigenvalues, eigvecs = spl.eig_banded(cov_matrix)
# sorted_indices = np.argsort(eigenvalues)[::-1]
# eigvecs = eigvecs[:, sorted_indices]

# scipy PCA 3
eigvals, eigvecs = spl.eigh(cov_matrix)
sorted_indices = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, sorted_indices]
# eigvals, eigvecs = eigvals * 1e6, eigvecs * 1e6
# eigvals, eigvecs = eigvals.astype(int), eigvecs.astype(int)
# eigvals, eigvecs = eigvals.astype(float), eigvecs.astype(float)
# eigvals, eigvecs = eigvals / 1e6, eigvecs / 1e6


# --- Compare matrices ---
print('\t\tCOV FUNCTION mean/std')
print('True:     \t', COV_MATRIX_TRUE.mean(), '\t', COV_MATRIX_TRUE.std(ddof=1))
print('Predicted:\t', cov_matrix.mean(), '\t', cov_matrix.std(ddof=1))
print()
print('\t\tPCACOV FUNCTION mean/std')
print('True:     \t', PCACOV_MATRIX_TRUE.mean(), '\t', PCACOV_MATRIX_TRUE.std(ddof=1))
print('Predicted:\t', eigvecs.mean(), '\t', eigvecs.std(ddof=1))
