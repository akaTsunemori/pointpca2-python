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


def compute_features(attA, attB, idA, idB,  searchSize):
    local_feats = np.full((attA.shape[0], 42), np.nan)
    for i in range(attA.shape[0]):
        dataA = attA[idA[i, 0:searchSize], :]
        dataB = attB[idB[i, 0:searchSize], :]
        geoA = dataA[:, 0:3]
        texA = dataA[:, 3:6]
        geoB = dataB[:, 0:3]
        texB = dataB[:, 3:6]
        covMatrixA = np.cov(geoA, rowvar=False, ddof=1)
        if np.sum(~np.isfinite(covMatrixA)) >= 1:
            eigvecsA = np.full((3, 3), np.nan)
        else:
            _, eigvecsA = np.linalg.eigh(covMatrixA)
            if eigvecsA.shape[1] != 3:
                eigvecsA = np.full((3, 3), np.nan)
        geoA_prA = (geoA - np.nanmean(geoA, axis=0)) @ eigvecsA
        geoB_prA = (geoB - np.nanmean(geoA, axis=0)) @ eigvecsA
        meanA = np.nanmean(np.array([geoA_prA, texA]), axis=0)
        meanB = np.nanmean(np.array([geoB_prA, texB]), axis=0)
        devmeanA = np.array([geoA_prA, texA]) - meanA
        devmeanB = np.array([geoB_prA, texB]) - meanB
        varA = np.nanmean(np.square(devmeanA), axis=0)
        varB = np.nanmean(np.square(devmeanB), axis=0)
        covAB = np.mean(devmeanA * devmeanB)
        covMatrixB = np.cov(geoB_prA, rowvar=False, ddof=1)
        if np.sum(~np.isfinite(covMatrixB)) >= 1:
            eigvecsB = np.full((3, 3), np.nan)
        else:
            _, eigvecsB = np.linalg.eigh(covMatrixB)
            if eigvecsB.shape[1] != 3:
                eigvecsB = np.full((3, 3), np.nan)
        meanA = meanA.flatten()
        meanB = meanB.flatten()
        varA = varA.flatten()
        varB = varB.flatten()
        # The dimensions for varA, varB, meanA and meanB
        # are wrong. Needs to be checked and fixed.
        local_feats[i, :] = np.concatenate([
            geoA_prA[0, :],          # 1-3
            geoB_prA[0, :],          # 4-6
            meanA[3:6],              # 7-9
            meanB[:5],               # 10-15
            varA,                    # 16-21
            varB,                    # 22-27
            [covAB],                 # 28-34
            eigvecsB[:, 0],          # 35-37
            eigvecsB[:, 1],          # 37-39
            eigvecsB[:, 2]           # 40-42
        ])
    return local_feats


def compute_predictors():
    pass


class PointPCA2:
    def __init__(self, 
            ref: o3d.geometry.PointCloud, 
            test: o3d.geometry.PointCloud) -> None:
        pass


# Load PCs
pc1 = o3d.io.read_point_cloud('pc1.ply')
pc2 = o3d.io.read_point_cloud('pc2.ply')

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
print(lfeats)
