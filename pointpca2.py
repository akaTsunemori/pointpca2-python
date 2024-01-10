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
        local_feats[i, :] = np.concatenate([
            geoA_prA[0, :],          # 1-3
            geoB_prA[0, :],          # 4-6
            meanA[3:],               # 7-9
            meanB,                   # 10-15
            varA,                    # 16-21
            varB,                    # 22-27
            covAB,                   # 28-34
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


# Create a sample point cloud with colors
points = np.random.rand(100, 3)
points[0] = points[1] # Force duplicated points
points[20] = points[21]
points *= 1000
colors = np.random.rand(100, 3)
colors *= 255
colors = colors.astype(int)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Duplicate Merging testing
print('Duplicate merging testing')
pcOut = pc_duplicate_merging(point_cloud)
pcOut_points = np.asarray(pcOut.points)
pcOut_colors = np.asarray(pcOut.colors)
print(pcOut_points[:10])
print(pcOut_colors[:10])

# RGB to YUV testing
print('RGB to YUV testing')
rgb_array = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
print(rgb_to_yuv(rgb_array))

# Perform KNN Search
print('KNN Search testing')
# searchSize = 81
searchSize = 3
# ref = o3d.io.read_point_cloud('flowerpot.ply')
# test = o3d.io.read_point_cloud('flowerpot_level_7.ply')
# ref_np = np.asarray(ref.points)
idA, distA = knnsearch(pcOut_points, pcOut_points, searchSize)
print(idA)
print(distA)

