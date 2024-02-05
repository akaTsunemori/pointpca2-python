import open3d as o3d
import pymeshlab
import tempfile
from contextlib import redirect_stdout
import io
import os


def safe_read_point_cloud(path: str):
    def get_suitable_version_path(ply_path, temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_filepath = os.path.join(temp_dir, "temp.ply")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(ply_path)
        ms.save_current_mesh(temp_ply_filepath)
        return temp_ply_filepath

    with tempfile.TemporaryDirectory() as temp_dir:
        with io.StringIO() as buf, redirect_stdout(buf):
            pc = o3d.io.read_point_cloud(path)
            output = buf.getvalue()
        if "Read PLY failed" in output:
            temp_ply = get_suitable_version_path(path, temp_dir)
            pc = o3d.io.read_point_cloud(temp_ply)
        return pc
