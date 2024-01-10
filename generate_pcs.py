import open3d as o3d
import numpy as np

# Generate sample point clouds with colors
# pc1
points = np.random.rand(100, 3)
points[0] = points[1] # Force duplicated points
points[20] = points[21]
points *= 1000
colors = np.random.rand(100, 3)
colors *= 255
colors = colors.astype(int)
pc1 = o3d.geometry.PointCloud()
pc1.points = o3d.utility.Vector3dVector(points)
pc1.colors = o3d.utility.Vector3dVector(colors)
# pc2
points = np.random.rand(100, 3)
points[0] = points[1] # Force duplicated points
points[20] = points[21]
points *= 1000
colors = np.random.rand(100, 3)
colors *= 255
colors = colors.astype(int)
pc2 = o3d.geometry.PointCloud()
pc2.points = o3d.utility.Vector3dVector(points)
pc2.colors = o3d.utility.Vector3dVector(colors)

# Save PCs
o3d.io.write_point_cloud('pc1.ply', pc1)
o3d.io.write_point_cloud('pc2.ply', pc2)
