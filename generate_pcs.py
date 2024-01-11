import open3d as o3d
import numpy as np

# Generate sample point clouds with colors
# pc1
points = np.random.rand(100, 3)
points[0] = points[1] # Force duplicated points
points[20] = points[21]
points *= 100
colors = np.random.rand(100, 3)
colors *= 255
colors = colors.astype(int)
pc1 = o3d.geometry.PointCloud()
pc1.points = o3d.utility.Vector3dVector(points)
pc1.colors = o3d.utility.Vector3dVector(colors)
# pc2
for i in range(len(points)):
    points[i] = points[i] + np.random.randint(-10, 10) / 100 * points[i]
for i in range(len(colors)):
    noise = np.random.randint(-10, 11) / 100 * colors[i]
    for j in range(len(noise)):
        noise[j] = int(noise[j])
        noise[j] = min(255, noise[j])
    colors[i] = colors[i] + noise
pc2 = o3d.geometry.PointCloud()
pc2.points = o3d.utility.Vector3dVector(points)
pc2.colors = o3d.utility.Vector3dVector(colors)

# Save PCs
o3d.io.write_point_cloud('pc3.ply', pc1)
o3d.io.write_point_cloud('pc4.ply', pc2)
