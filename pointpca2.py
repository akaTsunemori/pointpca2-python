import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.linalg import svd
from os.path import exists
from utils import safe_read_point_cloud


SEARCH_SIZE = 81
PREDICTORS_NUMBER = 40
EPS = np.finfo(float).eps


def denormalize_rgb(rgb):
    return (rgb * 255).astype(np.uint8)


def sort_pc(points, colors):
    pc = np.concatenate((points, colors), axis=1)
    py_list = pc.tolist()
    py_list.sort()
    pc = np.asarray(py_list, dtype=np.double)
    return pc[:, :3], pc[:, 3:]


def load_pc(path):
    if not exists(path):
        raise Exception('Path does not exist!')
    # pc = o3d.io.read_point_cloud(path)
    pc = safe_read_point_cloud(path)
    points = np.asarray(pc.points, dtype=np.double)
    colors = np.asarray(pc.colors, dtype=np.double)
    if not pc.has_colors():
        colors = np.full(points.shape, 0)
    colors = denormalize_rgb(colors)
    # points, colors = sort_pc(points, colors)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def pc_duplicate_merging(pc_input):
    pc_geometry = np.asarray(pc_input.points)
    pc_colors = np.asarray(pc_input.colors)
    unique_points = np.unique(pc_geometry, axis=0)
    if pc_geometry.shape[0] != unique_points.shape[0] and pc_colors.shape[0] != 0:
        points_sorted, colors_sorted = sort_pc(pc_geometry, pc_colors)
        diff = np.diff(points_sorted, axis=0)
        unique_indices = np.where(np.any(diff != 0, axis=1))[0] + 1
        id = np.concatenate(([0], unique_indices, [len(points_sorted)]))
        colors = np.zeros((len(id) - 1, 3))
        for j in range(len(id)-1):
            colors[j, :] = \
                np.round(np.mean(colors_sorted[id[j]:id[j+1], :], axis=0))
        id = id[:-1]
        unique_points = points_sorted[id, :]
        pc_colors = colors
    pc_output = o3d.geometry.PointCloud()
    pc_output.points = o3d.utility.Vector3dVector(unique_points)
    if pc_colors.shape[0] != 0:
        pc_output.colors = o3d.utility.Vector3dVector(pc_colors)
    return pc_output


def rgb_to_yuv(rgb):
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    c = np.array([[ 0.2126,  0.7152,  0.0722],
                  [-0.1146, -0.3854,  0.5000],
                  [ 0.5000, -0.4542, -0.0468]])
    o = np.array([0, 128, 128])
    y = c[0, 0]*r + c[0, 1]*g + c[0, 2]*b + o[0]
    u = c[1, 0]*r + c[1, 1]*g + c[1, 2]*b + o[1]
    v = c[2, 0]*r + c[2, 1]*g + c[2, 2]*b + o[2]
    yuv = np.column_stack(
        (np.round(y).astype(np.uint8),
         np.round(u).astype(np.uint8),
         np.round(v).astype(np.uint8)))
    return yuv


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
        geometry_A = search_data_A[:, :3]
        texture_A  = search_data_A[:, 3:6]
        geometry_B = search_data_B[:, :3]
        texture_B  = search_data_B[:, 3:6]
        eigenvectors_A = compute_eigenvectors(geometry_A)
        projection_AA = \
            (geometry_A - np.nanmean(geometry_A, axis=0)) @ eigenvectors_A
        projection_BA = \
            (geometry_B - np.nanmean(geometry_A, axis=0)) @ eigenvectors_A
        mean_A = np.nanmean(np.concatenate(
            (projection_AA, texture_A), axis=1), axis=0)
        mean_B = np.nanmean(np.concatenate(
            (projection_BA, texture_B), axis=1), axis=0)
        mean_deviation_A = np.concatenate(
            (projection_AA, texture_A), axis=1) - mean_A
        mean_deviation_B = np.concatenate(
            (projection_BA, texture_B), axis=1) - mean_B
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
    projection_AA             = local_features[:, 0:3]
    projection_BA             = local_features[:, 3:6]
    texture_mean_A            = local_features[:, 6:9]
    geometry_mean_B           = local_features[:, 9:12]
    texture_mean_B            = local_features[:, 12:15]
    geometry_variance_A       = local_features[:, 15:18]
    geometry_variance_B       = local_features[:, 21:24]
    texture_variance_A        = local_features[:, 18:21]
    texture_variance_B        = local_features[:, 24:27]
    geometry_covariance_AB    = local_features[:, 27:30]
    texture_covariance_AB     = local_features[:, 30:33]
    geometry_eigenvectors_B_x = local_features[:, 33:36]
    geometry_eigenvectors_B_y = local_features[:, 36:39]
    geometry_eigenvectors_B_z = local_features[:, 39:42]
    predictors = np.full((local_features.shape[0], 40), np.nan)
    # Textural predictors
    predictors[:, 0:3] = relative_difference(
        texture_mean_A, texture_mean_B)
    predictors[:, 3:6] = relative_difference(
        texture_variance_A, texture_variance_B)
    predictors[:, 6:9] = \
         np.abs(np.sqrt(texture_variance_A) * np.sqrt(texture_variance_B) - texture_covariance_AB) / \
        (np.sqrt(texture_variance_A) * np.sqrt(texture_variance_B) + EPS)
    predictors[:, 9] = relative_difference(
        np.sum(texture_variance_A, axis=1), np.sum(texture_variance_B, axis=1))
    predictors[:, 10] = relative_difference(
        np.prod(texture_variance_A, axis=1) ** (1 / 3),
        np.prod(texture_variance_B, axis=1) ** (1 / 3))
    predictors[:, 11] = relative_difference(
        -np.sum(texture_variance_A * np.log(texture_variance_A + EPS), axis=1),
        -np.sum(texture_variance_B * np.log(texture_variance_B + EPS), axis=1))
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
    predictors[:, 21] = np.sqrt(np.sum(geometry_mean_B ** 2, axis=1))
    predictors[:, 22:24] = np.abs(geometry_mean_B[:, 1:3])
    predictors[:, 24:27] = relative_difference(geometry_variance_A, geometry_variance_B)
    predictors[:, 27:30] = np.abs(
         np.sqrt(geometry_variance_A) * np.sqrt(geometry_variance_B) - geometry_covariance_AB) / \
        (np.sqrt(geometry_variance_A) * np.sqrt(geometry_variance_B) + EPS)
    predictors[:, 30] = relative_difference(
        np.prod(geometry_variance_A, axis=1) ** (1 / 3),
        np.prod(geometry_variance_B, axis=1) ** (1 / 3))
    predictors[:, 31] = relative_difference(
        -np.sum(geometry_variance_A * np.log(geometry_variance_A + EPS), axis=1),
        -np.sum(geometry_variance_B * np.log(geometry_variance_B + EPS), axis=1))
    predictors[:, 32] = relative_difference(
        (geometry_variance_A[:, 0] - geometry_variance_A[:, 2]) / geometry_variance_A[:, 0],
        (geometry_variance_B[:, 0] - geometry_variance_B[:, 2]) / geometry_variance_B[:, 0])
    predictors[:, 33] = relative_difference(
        (geometry_variance_A[:, 1] - geometry_variance_A[:, 2]) / geometry_variance_A[:, 0],
        (geometry_variance_B[:, 1] - geometry_variance_B[:, 2]) / geometry_variance_B[:, 0])
    predictors[:, 34] = relative_difference(
        (geometry_variance_A[:, 0] - geometry_variance_A[:, 1]) / geometry_variance_A[:, 0],
        (geometry_variance_B[:, 0] - geometry_variance_B[:, 1]) / geometry_variance_B[:, 0])
    predictors[:, 35] = relative_difference(
        geometry_variance_A[:, 2] / np.sum(geometry_variance_A, axis=1),
        geometry_variance_B[:, 2] / np.sum(geometry_variance_B, axis=1))
    predictors[:, 36] = relative_difference(
        geometry_variance_A[:, 2] / geometry_variance_A[:, 0],
        geometry_variance_B[:, 2] / geometry_variance_B[:, 0])
    predictors[:, 37] = 1 - 2 * np.arccos(
         np.abs(np.sum(np.array([0, 1, 0]) * geometry_eigenvectors_B_y, axis=1) /
        (np.sqrt(np.sum(np.array([0, 1, 0]) ** 2)) * np.sqrt(np.sum(geometry_eigenvectors_B_y ** 2, axis=1))))) / \
         np.pi
    predictors[:, 38] = 1 - np.sum(
        np.tile([1, 0, 0], (geometry_eigenvectors_B_x.shape[0], 1)) * geometry_eigenvectors_B_x, axis=1)
    predictors[:, 39] = 1 - np.sum(
        np.tile([0, 0, 1], (geometry_eigenvectors_B_z.shape[0], 1)) * geometry_eigenvectors_B_z, axis=1)
    return predictors


def pool_across_samples(samples):
    samples = samples[np.isfinite(samples)]
    pooled_samples = np.nanmean(samples.real)
    return pooled_samples


def lc_pointpca(path_to_reference, path_to_test):
    pc_A = load_pc(path_to_reference)
    pc_B = load_pc(path_to_test)
    pc_A = pc_duplicate_merging(pc_A)
    pc_B = pc_duplicate_merging(pc_B)
    geometry_A = np.asarray(pc_A.points, dtype=np.double)
    texture_A = rgb_to_yuv(np.asarray(pc_A.colors))
    geometry_B = np.asarray(pc_B.points, dtype=np.double)
    texture_B = rgb_to_yuv(np.asarray(pc_B.colors))
    _, knn_indices_A = knnsearch(geometry_A, geometry_A)
    _, knn_indices_B = knnsearch(geometry_B, geometry_A)
    attributes_A = np.concatenate([geometry_A, texture_A], axis=1)
    attributes_B = np.concatenate([geometry_B, texture_B], axis=1)
    local_features = compute_features(
        attributes_A, attributes_B, knn_indices_A, knn_indices_B)
    predictors = compute_predictors(local_features)
    lcpointpca = np.zeros(PREDICTORS_NUMBER)
    for i in range(PREDICTORS_NUMBER):
        lcpointpca[i] = pool_across_samples(predictors[:, i])
    for i in lcpointpca:
        print(i)
    return lcpointpca


if __name__ == '__main__':
    lc_pointpca(
        'path_to_ref.ply',
        'path_to_test.ply')
