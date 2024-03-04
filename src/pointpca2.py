import numpy as np
from os.path import exists
from scipy.spatial import KDTree
from scipy.linalg import svd
from utils import safe_read_point_cloud


SEARCH_SIZE = 81
PREDICTORS_NUMBER = 40
EPS = np.finfo(np.double).eps


def load_point_cloud(path):
    if not exists(path):
        raise Exception('Path does not exist.')
    pc = safe_read_point_cloud(path)
    points = np.asarray(pc.points, dtype=np.double)
    colors = np.asarray(pc.colors, dtype=np.double)
    if not pc.has_colors():
        colors = np.zeros(points.shape)
    return points, colors


def denormalize_rgb(rgb):
    return np.rint(rgb * 255).astype(np.uint)


def sort_pc(points, colors):
    pc_concat = np.concatenate((points, colors), axis=1)
    indices = np.lexsort(
        [pc_concat[:, col] for col in range(pc_concat.shape[1]-1, -1, -1)],
        axis=0)
    sorted_pc = pc_concat[indices]
    return sorted_pc[:, :3], sorted_pc[:, 3:]


def duplicate_merging(points, colors):
    if colors.shape[0] == 0:
        return points, colors
    points_colors_map = dict()
    for i in range(points.shape[0]):
        point = tuple(points[i])
        if point not in points_colors_map:
            points_colors_map[point] = []
        points_colors_map[point].append(colors[i])
    rows_num = len(points_colors_map)
    points_merged = np.empty((rows_num, 3), dtype=np.double)
    colors_merged = np.empty((rows_num, 3), dtype=np.double)
    for i, key in enumerate(points_colors_map):
        points_merged[i] = key
        colors_mean = np.mean(points_colors_map[key], axis=0)
        colors_merged[i] = colors_mean
    colors_merged = np.rint(colors_merged).astype(np.uint)
    return points_merged, colors_merged


def rgb_to_yuv(rgb):
    coefficients = np.array([
        [ 0.2126,  0.7152,  0.0722],
        [-0.1146, -0.3854,  0.5000],
        [ 0.5000, -0.4542, -0.0468]])
    offset = np.array([0, 128, 128])
    yuv = np.tensordot(rgb, coefficients, axes=[1, 1])
    yuv += offset
    yuv = np.rint(yuv).astype(np.uint)
    return yuv


def decimate_array(arr, decimation_factor):
    return arr[::decimation_factor]


def preprocess_point_cloud(points, colors, decimation_factor):
    if points.shape != colors.shape:
        raise Exception('Points and colors must have the same shape.')
    colors = denormalize_rgb(colors)
    points, colors = duplicate_merging(points, colors)
    points, colors = sort_pc(points, colors)
    colors = rgb_to_yuv(colors)
    if decimation_factor is not None:
        points = decimate_array(points, decimation_factor)
        colors = decimate_array(colors, decimation_factor)
    return points, colors


def knnsearch(va, vb):
    kdtree = KDTree(va)
    distances, indices = kdtree.query(vb, k=SEARCH_SIZE, p=2)
    return distances, indices


def svd_sign_correction(u, v, u_based_decision=True):
    if u_based_decision:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs[np.newaxis, :]
    v *= signs[:, np.newaxis]
    return u, v


def pcacov(covariance_matrix):
    U, S, Vt = svd(covariance_matrix, full_matrices=False, check_finite=False)
    U_corrected, Vt_corrected = \
        svd_sign_correction(U, Vt, u_based_decision=False)
    return U_corrected


def compute_eigenvectors(matrix):
    covariance_matrix = np.cov(matrix, rowvar=False, ddof=1, dtype=np.double)
    if not np.all(np.isfinite(covariance_matrix)):
        eigenvectors = np.full((3, 3), np.nan)
    else:
        eigenvectors = pcacov(covariance_matrix)
        if eigenvectors.shape[1] != 3:
            eigenvectors = np.full((3, 3), np.nan)
    return eigenvectors


def compute_features(attributes_A, attributes_B, knn_indices_A, knn_indices_B):
    local_features = np.full((attributes_A.shape[0], 42), np.nan)
    for i in range(attributes_A.shape[0]):
        search_data_A = attributes_A[knn_indices_A[i, :SEARCH_SIZE], :]
        search_data_B = attributes_B[knn_indices_B[i, :SEARCH_SIZE], :]
        points_A = search_data_A[:, :3]
        colors_A = search_data_A[:, 3:6]
        points_B = search_data_B[:, :3]
        colors_B = search_data_B[:, 3:6]
        eigenvectors_A = compute_eigenvectors(points_A)
        projection_AA = \
            (points_A - np.nanmean(points_A, axis=0)) @ eigenvectors_A
        projection_BA = \
            (points_B - np.nanmean(points_A, axis=0)) @ eigenvectors_A
        mean_A = np.nanmean(np.concatenate(
            (projection_AA, colors_A), axis=1), axis=0)
        mean_B = np.nanmean(np.concatenate(
            (projection_BA, colors_B), axis=1), axis=0)
        mean_deviation_A = np.concatenate(
            (projection_AA, colors_A), axis=1) - mean_A
        mean_deviation_B = np.concatenate(
            (projection_BA, colors_B), axis=1) - mean_B
        variance_A = np.nanmean(mean_deviation_A**2, axis=0)
        variance_B = np.nanmean(mean_deviation_B**2, axis=0)
        covariance_AB = np.nanmean(mean_deviation_A * mean_deviation_B, axis=0)
        eigenvectors_B = compute_eigenvectors(projection_BA)
        local_features[i, :] = np.concatenate((projection_AA[0],           # 1-3
                                               projection_BA[0],           # 4-6
                                               mean_A[3:6],                # 7-9
                                               mean_B,                     # 10-15
                                               variance_A,                 # 16-21
                                               variance_B,                 # 22-27
                                               covariance_AB,              # 28-34
                                               eigenvectors_B[:, 0],       # 35-37
                                               eigenvectors_B[:, 1],       # 37-39
                                               eigenvectors_B[:, 2]))      # 40-42
    return local_features


def relative_difference(X, Y):
    return 1 - np.abs(X - Y) / (np.abs(X) + np.abs(Y) + EPS)


def compute_predictors(local_features):
    projection_AA           = local_features[:, 0:3]
    projection_BA           = local_features[:, 3:6]
    colors_mean_A           = local_features[:, 6:9]
    points_mean_B           = local_features[:, 9:12]
    colors_mean_B           = local_features[:, 12:15]
    points_variance_A       = local_features[:, 15:18]
    points_variance_B       = local_features[:, 21:24]
    colors_variance_A       = local_features[:, 18:21]
    colors_variance_B       = local_features[:, 24:27]
    points_covariance_AB    = local_features[:, 27:30]
    colors_covariance_AB    = local_features[:, 30:33]
    points_eigenvectors_B_x = local_features[:, 33:36]
    points_eigenvectors_B_y = local_features[:, 36:39]
    points_eigenvectors_B_z = local_features[:, 39:42]
    predictors = np.full((local_features.shape[0], 40), np.nan)
    # Textural predictors
    predictors[:, 0:3] = relative_difference(
        colors_mean_A, colors_mean_B)
    predictors[:, 3:6] = relative_difference(
        colors_variance_A, colors_variance_B)
    predictors[:, 6:9] = \
         np.abs(np.sqrt(colors_variance_A) * np.sqrt(colors_variance_B) - colors_covariance_AB) / \
        (np.sqrt(colors_variance_A) * np.sqrt(colors_variance_B) + EPS)
    predictors[:, 9] = relative_difference(
        np.sum(colors_variance_A, axis=1), np.sum(colors_variance_B, axis=1))
    predictors[:, 10] = relative_difference(
        np.prod(colors_variance_A, axis=1) ** (1 / 3),
        np.prod(colors_variance_B, axis=1) ** (1 / 3))
    predictors[:, 11] = relative_difference(
        -np.sum(colors_variance_A * np.log(colors_variance_A + EPS), axis=1),
        -np.sum(colors_variance_B * np.log(colors_variance_B + EPS), axis=1))
    # Geometric predictors
    predictors[:, 12] = np.sqrt(
        np.sum((projection_BA - projection_AA) ** 2, axis=1))
    predictors[:, 13] = np.abs(np.sum(
        (projection_BA - projection_AA) * np.tile([1, 0, 0],
        (projection_AA.shape[0], 1)), axis=1))
    predictors[:, 14] = np.abs(np.sum(
        (projection_BA - projection_AA) * np.tile([0, 1, 0],
        (projection_AA.shape[0], 1)), axis=1))
    predictors[:, 15] = np.abs(np.sum(
        (projection_BA - projection_AA) * np.tile([0, 0, 1],
        (projection_AA.shape[0], 1)), axis=1))
    predictors[:, 16:18] = np.abs(projection_AA[:, 1:3])
    predictors[:, 18] = np.sqrt(np.sum(projection_BA**2, axis=1))
    predictors[:, 19:21] = np.abs(projection_BA[:, 1:3])
    predictors[:, 21] = np.sqrt(np.sum(points_mean_B ** 2, axis=1))
    predictors[:, 22:24] = np.abs(points_mean_B[:, 1:3])
    predictors[:, 24:27] = relative_difference(points_variance_A, points_variance_B)
    predictors[:, 27:30] = np.abs(
         np.sqrt(points_variance_A) * np.sqrt(points_variance_B) - points_covariance_AB) / \
        (np.sqrt(points_variance_A) * np.sqrt(points_variance_B) + EPS)
    predictors[:, 30] = relative_difference(
        np.prod(points_variance_A, axis=1) ** (1 / 3),
        np.prod(points_variance_B, axis=1) ** (1 / 3))
    predictors[:, 31] = relative_difference(
        -np.sum(points_variance_A * np.log(points_variance_A + EPS), axis=1),
        -np.sum(points_variance_B * np.log(points_variance_B + EPS), axis=1))
    predictors[:, 32] = relative_difference(
        (points_variance_A[:, 0] - points_variance_A[:, 2]) / points_variance_A[:, 0],
        (points_variance_B[:, 0] - points_variance_B[:, 2]) / points_variance_B[:, 0])
    predictors[:, 33] = relative_difference(
        (points_variance_A[:, 1] - points_variance_A[:, 2]) / points_variance_A[:, 0],
        (points_variance_B[:, 1] - points_variance_B[:, 2]) / points_variance_B[:, 0])
    predictors[:, 34] = relative_difference(
        (points_variance_A[:, 0] - points_variance_A[:, 1]) / points_variance_A[:, 0],
        (points_variance_B[:, 0] - points_variance_B[:, 1]) / points_variance_B[:, 0])
    predictors[:, 35] = relative_difference(
        points_variance_A[:, 2] / np.sum(points_variance_A, axis=1),
        points_variance_B[:, 2] / np.sum(points_variance_B, axis=1))
    predictors[:, 36] = relative_difference(
        points_variance_A[:, 2] / points_variance_A[:, 0],
        points_variance_B[:, 2] / points_variance_B[:, 0])
    predictors[:, 37] = 1 - 2 * np.arccos(
         np.abs(np.sum(np.array([0, 1, 0]) * points_eigenvectors_B_y, axis=1) /
        (np.sqrt(np.sum(np.array([0, 1, 0]) ** 2)) * np.sqrt(np.sum(points_eigenvectors_B_y ** 2, axis=1))))) / \
         np.pi
    predictors[:, 38] = 1 - np.sum(
        np.tile([1, 0, 0], (points_eigenvectors_B_x.shape[0], 1)) * points_eigenvectors_B_x, axis=1)
    predictors[:, 39] = 1 - np.sum(
        np.tile([0, 0, 1], (points_eigenvectors_B_z.shape[0], 1)) * points_eigenvectors_B_z, axis=1)
    return predictors


def pool_across_samples(samples):
    samples = samples[np.isfinite(samples)]
    pooled_samples = np.nanmean(samples.real)
    return pooled_samples


def pointpca2(path_to_reference, path_to_test, decimation_factor=None):
    points_A, colors_A = load_point_cloud(path_to_reference)
    points_B, colors_B = load_point_cloud(path_to_test)
    points_A, colors_A = preprocess_point_cloud(points_A, colors_A, decimation_factor)
    points_B, colors_B = preprocess_point_cloud(points_B, colors_B, decimation_factor)
    _, knn_indices_A = knnsearch(points_A, points_A)
    _, knn_indices_B = knnsearch(points_B, points_A)
    attributes_A = np.concatenate([points_A, colors_A], axis=1)
    attributes_B = np.concatenate([points_B, colors_B], axis=1)
    local_features = compute_features(
        attributes_A, attributes_B, knn_indices_A, knn_indices_B)
    predictors = compute_predictors(local_features)
    lcpointpca = np.zeros(PREDICTORS_NUMBER)
    for i in range(PREDICTORS_NUMBER):
        lcpointpca[i] = pool_across_samples(predictors[:, i])
    return lcpointpca


if __name__ == '__main__':
    pointpca2(
        'path_to_ref.ply',
        'path_to_test.ply')
