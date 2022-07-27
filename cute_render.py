import numpy as np
import open3d as o3d
from tqdm import tqdm

import utility

def cute_render(args):
    textures, positions, labels = utility.read(args, ['textures', 'positions', 'labels'])
    texture_lib = utility.read_texture_lib(args)
    n_cutes = textures.shape[0]

    model = o3d.geometry.TriangleMesh()
    lib_textures_list = [o3d.geometry.Image(im) for im in texture_lib]
    ori_textures_list = []
    material_id_list = []
    for i in tqdm(range(n_cutes), desc="Generating cutes", leave=True):
        box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True, map_texture_to_each_face=True)
        loc = np.array(box.vertices) * args.cube_size
        loc[:,0] += positions[i, 0]
        loc[:,1] += positions[i, 1]
        loc[:,2] += positions[i, 2]
        box.vertices = o3d.utility.Vector3dVector(loc)
        triangle_uvs = np.vstack([np.asarray(model.triangle_uvs), np.asarray(box.triangle_uvs)])
        model += box
        model.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)


        t = np.asarray(textures[i].T, dtype=np.float32, order='c')
        ori_textures_list.append(o3d.geometry.Image(t))
        material_id_list.append(labels[i] if not args.RENDER_RAST else i)
    print(f'total vertices of the model: {len(model.vertices)}')
    model.textures = lib_textures_list if not args.RENDER_RAST else ori_textures_list
    model.triangle_material_ids = o3d.utility.IntVector(np.asarray(material_id_list, dtype=np.int32).repeat(12))
    o3d.visualization.draw_geometries([model])