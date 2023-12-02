import open3d as o3d

pcd = o3d.io.read_point_cloud("../results/temple.pcd")
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultUnlit'
mat.point_size = 2.0
o3d.visualization.draw([{'name': 'pcd', 'geometry': pcd, 'material': mat}], bg_color=(0, 0, 0, 0))
