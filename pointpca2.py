import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import KDTree
from sklearn.decomposition import IncrementalPCA
from os.path import exists

# from utils import safe_read_point_cloud


searchSize = 81 # Default = 81
numPreds = 40


def sort_pc(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    pc = np.concatenate((points, colors), axis=1)
    py_list = pc.tolist()
    py_list.sort()
    pc = np.asarray(py_list, dtype=np.double)
    return pc[:, :3], pc[:, 3:]


def load_pc(path):
    if not exists(path):
        raise Exception('Path does not exist!')
    # pc = safe_read_point_cloud(path)
    pc = o3d.io.read_point_cloud(path)
    points = np.asarray(pc.points, dtype=np.double)
    colors = np.asarray(pc.colors, dtype=np.double)
    points, colors = sort_pc(points, colors)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc


def pc_duplicate_merging(pcIn: o3d.geometry.PointCloud):
    geomIn = np.asarray(pcIn.points)
    colorsIn = np.asarray(pcIn.colors)
    vertices, ind_v = np.unique(geomIn, axis=0, return_index=True)
    if geomIn.shape[0] != vertices.shape[0]:
        # print('** Warning: Duplicated points found.')
        if colorsIn.shape[0] != 0:
            # print('** Color blending is applied.')
            vertices_sorted, colors_sorted = sort_pc(geomIn, colorsIn)
            d = np.diff(vertices_sorted, axis=0)
            sd = np.sum(np.abs(d), axis=1) > 0
            id = np.concatenate(
                ([0], np.where(sd)[0] + 1, [vertices_sorted.shape[0]]))
            colors = np.zeros((len(id) - 1, 3))
            for j in range(len(id)-1):
                colors[j, :] = np.round(
                    np.mean(colors_sorted[id[j]:id[j+1], :], axis=0))
            id = id[:-1]
            vertices = vertices_sorted[id, :]
            colorsIn = colors
    pcOut = o3d.geometry.PointCloud()
    pcOut.points = o3d.utility.Vector3dVector(vertices)
    if colorsIn.shape[0] != 0:
        pcOut.colors = o3d.utility.Vector3dVector(colorsIn)
    return pcOut


def denormalize_rgb(rgb):
    return (rgb * 255).astype(np.uint8)


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


def knnsearch(va: np.ndarray, vb: np.ndarray, search_size: int) -> np.ndarray:
    kdtree = KDTree(va)
    distances, indices = kdtree.query(vb, k=search_size, p=2)
    return distances, indices


def pca(M):
    pca = IncrementalPCA()
    pca.fit(M)
    eigvecs = pca.components_.T
    return eigvecs


def compute_features(attA, attB, idA, idB, searchSize):
    local_feats = np.full((attA.shape[0], 42), np.nan)
    for i in range(attA.shape[0]):
        dataA = attA[idA[i, :searchSize], :]
        dataB = attB[idB[i, :searchSize], :]
        geoA = dataA[:, :3]
        texA = dataA[:, 3:6]
        geoB = dataB[:, :3]
        texB = dataB[:, 3:6]
        covMatrixA = np.cov(geoA, rowvar=False, ddof=1, dtype=np.double)
        if not np.all(np.isfinite(covMatrixA)):
            eigvecsA = np.full((3, 3), np.nan)
        else:
            # eigvecsA = pcacov(covMatrixA)
            eigvecsA = pca(geoA)
            if eigvecsA.shape[1] != 3:
                eigvecsA = np.full((3, 3), np.nan)
        geoA_prA = (geoA - np.nanmean(geoA, axis=0)) @ eigvecsA
        geoB_prA = (geoB - np.nanmean(geoA, axis=0)) @ eigvecsA
        meanA = np.nanmean(np.concatenate((geoA_prA, texA), axis=1), axis=0)
        meanB = np.nanmean(np.concatenate((geoB_prA, texB), axis=1), axis=0)
        devmeanA = np.concatenate((geoA_prA, texA), axis=1) - meanA
        devmeanB = np.concatenate((geoB_prA, texB), axis=1) - meanB
        varA = np.nanmean(devmeanA**2, axis=0)
        varB = np.nanmean(devmeanB**2, axis=0)
        covAB = np.nanmean(devmeanA * devmeanB, axis=0)
        covMatrixB = np.cov(geoB_prA, rowvar=False, ddof=1, dtype=np.double)
        if not np.all(np.isfinite(covMatrixB)):
            eigvecsB = np.full((3, 3), np.nan)
        else:
            # eigvecsB = pcacov(covMatrixB)
            eigvecsB = pca(geoB_prA)
            if eigvecsB.shape[1] != 3:
                eigvecsB = np.full((3, 3), np.nan)
        local_feats[i, :] = np.concatenate((geoA_prA[0],          # 1-3
                                            geoB_prA[0],          # 4-6
                                            meanA[3:6],           # 7-9
                                            meanB,                # 10-15
                                            varA,                 # 16-21
                                            varB,                 # 22-27
                                            covAB,                # 28-34
                                            eigvecsB[:, 0],       # 35-37
                                            eigvecsB[:, 1],       # 37-39
                                            eigvecsB[:, 2]))      # 40-42
    return local_feats


def rel_diff(X, Y):
    return 1 - np.abs(X - Y) / (np.abs(X) + np.abs(Y) + np.finfo(float).eps)


def compute_predictors(lfeats):
    predNames = ['t_mu_y', 't_mu_u', 't_mu_v',
                 't_var_y', 't_var_u', 't_var_v',
                 't_cov_y', 't_cov_u', 't_cov_v',
                 't_varsum',
                 't_omnivariance',
                 't_entropy',
                 'g_pB2pA',
                 'g_vAB2plA_x', 'g_vAB2plA_y', 'g_vAB2plA_z',
                 'g_pA2plA_y', 'g_pA2plA_z',
                 'g_pB2cA',
                 'g_pB2plA_y', 'g_pB2plA_z',
                 'g_cB2cA',
                 'g_cB2plA_y', 'g_cB2plA_z',
                 'g_var_x', 'g_var_y', 'g_var_z',
                 'g_cov_x', 'g_cov_y', 'g_cov_z',
                 'g_omnivariance',
                 'g_entropy',
                 'g_anisotropy',
                 'g_planarity',
                 'g_linearity',
                 'g_surfaceVariation',
                 'g_sphericity',
                 'g_asim_y',
                 'g_paralellity_x', 'g_paralellity_z']
    pA = lfeats[:, 0:3]
    pB = lfeats[:, 3:6]
    tmeanA = lfeats[:, 6:9]
    gmeanB = lfeats[:, 9:12]
    tmeanB = lfeats[:, 12:15]
    gvarA = lfeats[:, 15:18]
    gvarB = lfeats[:, 21:24]
    tvarA = lfeats[:, 18:21]
    tvarB = lfeats[:, 24:27]
    gcovAB = lfeats[:, 27:30]
    tcovAB = lfeats[:, 30:33]
    geigvecB_x = lfeats[:, 33:36]
    geigvecB_y = lfeats[:, 36:39]
    geigvecB_z = lfeats[:, 39:42]
    # Initialization
    preds = np.full((lfeats.shape[0], 40), np.nan)
    # Textural predictors
    preds[:, 0:3] = rel_diff(tmeanA, tmeanB)
    preds[:, 3:6] = rel_diff(tvarA, tvarB)
    preds[:, 6:9] = np.abs(np.sqrt(tvarA) * np.sqrt(tvarB) - tcovAB) / \
        (np.sqrt(tvarA) * np.sqrt(tvarB) + np.finfo(float).eps)
    preds[:, 9] = rel_diff(np.sum(tvarA, axis=1), np.sum(tvarB, axis=1))
    preds[:, 10] = rel_diff(np.prod(tvarA, axis=1) **
                            (1 / 3), np.prod(tvarB, axis=1) ** (1 / 3))
    preds[:, 11] = rel_diff(-np.sum(tvarA * np.log(tvarA + np.finfo(float).eps), axis=1),
                            -np.sum(tvarB * np.log(tvarB + np.finfo(float).eps), axis=1))
    # Geometric predictors
    preds[:, 12] = np.sqrt(np.sum((pB - pA) ** 2, axis=1))
    preds[:, 13] = np.abs(
        np.sum((pB - pA) * np.tile([1, 0, 0], (pA.shape[0], 1)), axis=1))
    preds[:, 14] = np.abs(
        np.sum((pB - pA) * np.tile([0, 1, 0], (pA.shape[0], 1)), axis=1))
    preds[:, 15] = np.abs(
        np.sum((pB - pA) * np.tile([0, 0, 1], (pA.shape[0], 1)), axis=1))
    preds[:, 16:18] = np.abs(pA[:, 1:3])
    preds[:, 18] = np.sqrt(np.sum(pB**2, axis=1))
    preds[:, 19:21] = np.abs(pB[:, 1:3])
    preds[:, 21] = np.sqrt(np.sum(gmeanB ** 2, axis=1))
    preds[:, 22:24] = np.abs(gmeanB[:, 1:3])
    preds[:, 24:27] = rel_diff(gvarA, gvarB)
    preds[:, 27:30] = np.abs(np.sqrt(gvarA) * np.sqrt(gvarB) - gcovAB) / (
        np.sqrt(gvarA) * np.sqrt(gvarB) + np.finfo(float).eps)
    preds[:, 30] = rel_diff(np.prod(gvarA, axis=1) **
                            (1 / 3), np.prod(gvarB, axis=1) ** (1 / 3))
    preds[:, 31] = rel_diff(-np.sum(gvarA * np.log(gvarA + np.finfo(float).eps), axis=1),
                            -np.sum(gvarB * np.log(gvarB + np.finfo(float).eps), axis=1))
    preds[:, 32] = rel_diff((gvarA[:, 0] - gvarA[:, 2]) /
                            gvarA[:, 0], (gvarB[:, 0] - gvarB[:, 2]) / gvarB[:, 0])
    preds[:, 33] = rel_diff((gvarA[:, 1] - gvarA[:, 2]) /
                            gvarA[:, 0], (gvarB[:, 1] - gvarB[:, 2]) / gvarB[:, 0])
    preds[:, 34] = rel_diff((gvarA[:, 0] - gvarA[:, 1]) /
                            gvarA[:, 0], (gvarB[:, 0] - gvarB[:, 1]) / gvarB[:, 0])
    preds[:, 35] = rel_diff(
        gvarA[:, 2] / np.sum(gvarA, axis=1), gvarB[:, 2] / np.sum(gvarB, axis=1))
    preds[:, 36] = rel_diff(gvarA[:, 2] / gvarA[:, 0],
                            gvarB[:, 2] / gvarB[:, 0])
    preds[:, 37] = 1 - 2 * np.arccos(np.abs(np.sum(np.array([0, 1, 0]) * geigvecB_y, axis=1) / (
        np.sqrt(np.sum(np.array([0, 1, 0]) ** 2)) * np.sqrt(np.sum(geigvecB_y ** 2, axis=1))))) / np.pi
    preds[:, 38] = 1 - np.sum(np.tile([1, 0, 0],
                              (geigvecB_x.shape[0], 1)) * geigvecB_x, axis=1)
    preds[:, 39] = 1 - np.sum(np.tile([0, 0, 1],
                              (geigvecB_z.shape[0], 1)) * geigvecB_z, axis=1)
    return preds, predNames


def pool_across_samples(samples):
    samples = samples[np.isfinite(samples)]
    if samples.shape[0] == 0:
        return np.nan
    pooled_samples = np.nanmean(samples.real)
    return pooled_samples


def lc_pointpca(filenameRef, filenameDis):
    # Load PCs
    pc1 = load_pc(filenameRef)
    pc2 = load_pc(filenameDis)
    # pc_duplicate_merging
    # print('pc_duplicate_merging')
    pc1 = pc_duplicate_merging(pc1)
    pc2 = pc_duplicate_merging(pc2)
    # rgb_to_yuv
    # print('rgb_to_yuv')
    geoA = np.asarray(pc1.points, dtype=np.double)
    texA = rgb_to_yuv(
        denormalize_rgb(np.asarray(pc1.colors)))
    geoB = np.asarray(pc2.points, dtype=np.double)
    texB = rgb_to_yuv(
        denormalize_rgb(np.asarray(pc2.colors)))
    # knnsearch
    # print('knnsearch')
    _, idA = knnsearch(geoA, geoA, searchSize)
    _, idB = knnsearch(geoB, geoA, searchSize)
    # compute_features
    # print('compute_features')
    attA = np.concatenate([geoA, texA], axis=1)
    attB = np.concatenate([geoB, texB], axis=1)
    lfeats = compute_features(attA, attB, idA, idB, searchSize)
    # compute_predictors
    # print('lfeats')
    preds, predNames = compute_predictors(lfeats)
    # pool_across_samples
    # print('lcpointpca')
    lcpointpca = np.zeros(numPreds)
    for i in range(numPreds):
        lcpointpca[i] = pool_across_samples(preds[:, i])
    # for val in lcpointpca:
    #     print(f'{val:.4f},')
    return lcpointpca


if __name__ == '__main__':
    lc_pointpca(
        "/home/arthurc/Documents/APSIPA/PVS/tmc13_amphoriskos_vox10_dec_geom01_text01_octree-predlift.ply",
        "/home/arthurc/Documents/APSIPA/references/amphoriskos_vox10.ply")
