import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


def pc_duplicate_merging(pcIn: o3d.geometry.PointCloud):
    geomIn = np.asarray(pcIn.points)
    colorsIn = np.asarray(pcIn.colors)
    vertices, ind_v = np.unique(geomIn, axis=0, return_index=True)
    if geomIn.shape[0] != vertices.shape[0]:
        print('Warning: Duplicated points found.')
        if colorsIn.shape[0] != 0:
            print('Color blending is applied.')
            ind_v = np.lexsort(geomIn.T)
            vertices_sorted = geomIn[ind_v]
            colors_sorted = colorsIn[ind_v]
            d = np.diff(vertices_sorted, axis=0)
            sd = np.sum(np.abs(d), axis=1) > 0
            id = np.concatenate(([0], np.where(sd)[0] + 1, [vertices_sorted.shape[0]]))
            colors = np.zeros((len(id) - 1, 3))
            for j in range(len(id)-1):
                colors[j, :] = np.round(np.mean(colors_sorted[id[j]:id[j+1], :], axis=0))
            id = id[:-1]
            vertices = vertices_sorted[id, :]
            colorsIn = colors
    pcOut = o3d.geometry.PointCloud()
    pcOut.points = o3d.utility.Vector3dVector(vertices)
    if colorsIn.shape[0] != 0:
        pcOut.colors = o3d.utility.Vector3dVector(colorsIn)
    return pcOut


def rgb_to_yuv(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)
    v = 0.877 * (r - y)
    yuv = np.column_stack((y, u, v))
    return yuv.astype(int)


def knnsearch(va: np.ndarray, vb: np.ndarray, search_size: int) -> np.ndarray:
    distances = []
    indices = []
    kdtree = KDTree(va)
    for pb in vb:
        distance, index = kdtree.query(pb, k=search_size, p=2)
        distances.append(distance)
        indices.append(index)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    return (distances, indices)


def compute_features(attA, attB, idA, idB, searchSize):
    local_feats = np.empty((attA.shape[0], 42))
    for i in range(attA.shape[0]):
        dataA = attA[idA[i, :searchSize], :]
        dataB = attB[idB[i, :searchSize], :]
        geoA = dataA[:, :3]
        texA = dataA[:, 3:6]
        geoB = dataB[:, :3]
        texB = dataB[:, 3:6]
        covMatrixA = np.cov(geoA, rowvar=False)
        if not np.all(np.isfinite(covMatrixA)):
            eigvecsA = np.full((3, 3), np.nan)
        else:
            _, eigvecsA = np.linalg.eigh(covMatrixA)
            if eigvecsA.shape[1] != 3:
                eigvecsA = np.full((3, 3), np.nan)
        geoA_prA = (geoA - np.mean(geoA, axis=0)) @ eigvecsA
        geoB_prA = (geoB - np.mean(geoA, axis=0)) @ eigvecsA
        meanA = np.mean(np.concatenate((geoA_prA, texA), axis=1), axis=0)
        meanB = np.mean(np.concatenate((geoB_prA, texB), axis=1), axis=0)
        devmeanA = np.concatenate((geoA_prA, texA), axis=1) - meanA
        devmeanB = np.concatenate((geoB_prA, texB), axis=1) - meanB
        varA = np.mean(devmeanA**2, axis=0)
        varB = np.mean(devmeanB**2, axis=0)
        covAB = np.mean(devmeanA * devmeanB, axis=0)
        covMatrixB = np.cov(geoB_prA, rowvar=False)
        if not np.all(np.isfinite(covMatrixB)):
            eigvecsB = np.full((3, 3), np.nan)
        else:
            _, eigvecsB = np.linalg.eigh(covMatrixB)
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
    preds = np.nan * np.ones((lfeats.shape[0], 40))
    # Textural predictors
    preds[:, 0:3] = rel_diff(tmeanA, tmeanB)
    preds[:, 3:6] = rel_diff(tvarA, tvarB)
    preds[:, 6:9] = np.abs(np.sqrt(tvarA) * np.sqrt(tvarB) - tcovAB) / (np.sqrt(tvarA) * np.sqrt(tvarB) + np.finfo(float).eps)
    preds[:, 9] = rel_diff(np.sum(tvarA, axis=1), np.sum(tvarB, axis=1))
    preds[:, 10] = rel_diff(np.prod(tvarA, axis=1) ** (1 / 3), np.prod(tvarB, axis=1) ** (1 / 3))
    preds[:, 11] = rel_diff(-np.sum(tvarA * np.log(tvarA + np.finfo(float).eps), axis=1),
                            -np.sum(tvarB * np.log(tvarB + np.finfo(float).eps), axis=1))
    # Geometric predictors
    preds[:, 12] = np.sqrt(np.sum((pB - pA) ** 2, axis=1))
    preds[:, 13] = np.abs(np.dot(pB - pA, np.array([1, 0, 0])))
    preds[:, 14] = np.abs(np.dot(pB - pA, np.array([0, 1, 0])))
    preds[:, 15] = np.abs(np.dot(pB - pA, np.array([0, 0, 1])))
    preds[:, 16:18] = np.abs(pA[:, 1:3])
    preds[:, 17] = np.abs(pA[:, 2])
    preds[:, 18] = np.sqrt(np.sum(pB ** 2, axis=1))
    preds[:, 19:21] = np.abs(pB[:, 1:3])
    preds[:, 20] = np.abs(pB[:, 2])
    preds[:, 21] = np.sqrt(np.sum(gmeanB ** 2, axis=1))
    preds[:, 22:24] = np.abs(gmeanB[:, 1:3])
    preds[:, 23] = np.abs(gmeanB[:, 2])
    preds[:, 24:27] = rel_diff(gvarA, gvarB)
    preds[:, 26:29] = np.abs(np.sqrt(gvarA) * np.sqrt(gvarB) - gcovAB) / (
            np.sqrt(gvarA) * np.sqrt(gvarB) + np.finfo(float).eps)
    preds[:, 29] = rel_diff(np.prod(gvarA, axis=1) ** (1 / 3), np.prod(gvarB, axis=1) ** (1 / 3))
    preds[:, 30] = rel_diff(-np.sum(gvarA * np.log(gvarA + np.finfo(float).eps), axis=1),
                            -np.sum(gvarB * np.log(gvarB + np.finfo(float).eps), axis=1))
    preds[:, 31] = rel_diff((gvarA[:, 0] - gvarA[:, 2]) / gvarA[:, 0], (gvarB[:, 0] - gvarB[:, 2]) / gvarB[:, 0])
    preds[:, 32] = rel_diff((gvarA[:, 1] - gvarA[:, 2]) / gvarA[:, 0], (gvarB[:, 1] - gvarB[:, 2]) / gvarB[:, 0])
    preds[:, 33] = rel_diff((gvarA[:, 0] - gvarA[:, 1]) / gvarA[:, 0], (gvarB[:, 0] - gvarB[:, 1]) / gvarB[:, 0])
    preds[:, 34] = rel_diff(gvarA[:, 2] / np.sum(gvarA, axis=1), gvarB[:, 2] / np.sum(gvarB, axis=1))
    preds[:, 35] = rel_diff(gvarA[:, 2] / gvarA[:, 0], gvarB[:, 2] / gvarB[:, 0])
    preds[:, 36] = 1 - 2 * np.arccos(np.abs(np.sum(np.array([0, 1, 0]) * geigvecB_y, axis=1) / (
        np.sqrt(np.sum(np.array([0, 1, 0]) ** 2)) * np.sqrt(np.sum(geigvecB_y ** 2, axis=1))))) / np.pi
    preds[:, 37] = 1 - np.sum(np.tile([1, 0, 0], (geigvecB_x.shape[0], 1)) * geigvecB_x, axis=1)
    preds[:, 38] = 1 - np.sum(np.tile([0, 0, 1], (geigvecB_z.shape[0], 1)) * geigvecB_z, axis=1)
    return preds, predNames


def pool_across_samples(samples):
    samples = samples[np.isfinite(samples)]
    if samples.shape[0] == 0:
        return np.nan
    pooled_samples = np.nanmean(samples.real)
    return pooled_samples


# Load PCs
pc1 = o3d.io.read_point_cloud('matlab_dump/pc1.ply')
pc2 = o3d.io.read_point_cloud('matlab_dump/pc2.ply')

# pc_duplicate_merging testing
print('pc_duplicate_merging')
pc1 = pc_duplicate_merging(pc1)
pc2 = pc_duplicate_merging(pc2)

# rgb_to_yuv testing
print('rgb_to_yuv')
geoA = np.asarray(pc1.points)
texA = rgb_to_yuv(np.asarray(pc1.colors))
geoB = np.asarray(pc2.points)
texB = rgb_to_yuv(np.asarray(pc2.colors))

# knnsearch testing
print('knnsearch')
# searchSize = 81
searchSize = 3
_, idA = knnsearch(geoA, geoA, searchSize)
_, idB = knnsearch(geoB, geoA, searchSize)

# compute_features testing
print('compute_features')
geoA = geoA.reshape(-1, 1) if geoA.ndim == 1 else geoA
texA = texA.reshape(-1, 1) if texA.ndim == 1 else texA
geoB = geoB.reshape(-1, 1) if geoB.ndim == 1 else geoB
texB = texB.reshape(-1, 1) if texB.ndim == 1 else texB
attA = np.concatenate([geoA, texA], axis=1)
attB = np.concatenate([geoB, texB], axis=1) 
lfeats = compute_features(attA, attB, idA, idB, searchSize)

# compute_predictors testing
preds, predNames = compute_predictors(lfeats)

# pool_across_samples testing
numPreds = 40
lcpointpca = np.zeros(numPreds)
for i in range(numPreds):
    lcpointpca[i] = pool_across_samples(preds[:, i])
print(*lcpointpca, sep='\n')