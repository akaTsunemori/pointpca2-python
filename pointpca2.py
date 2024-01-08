import numpy as np
import open3d as o3d


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


class PointPCA2:
    def __init__(self, 
            ref: o3d.geometry.PointCloud, 
            test: o3d.geometry.PointCloud) -> None:
        pass


# ref = o3d.io.read_point_cloud('flowerpot.ply')
# test = o3d.io.read_point_cloud('flowerpot_level_7.ply')


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
pcOut = pc_duplicate_merging(point_cloud)
pcOut_points = np.asarray(pcOut.points)
pcOut_colors = np.asarray(pcOut.colors)
print(pcOut_points)
print(pcOut_colors)

# RGB to YUV testing
rgb_array = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
print(rgb_to_yuv(rgb_array))
