import numpy as np
from tqdm import tqdm

import open3d as o3d
from cute_rasterization import cute_rasterization
import utility

def cute_voxelize(args):
    pcd = o3d.io.read_point_cloud(args.pcd_filename)
    print(pcd)

    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    n_pts = pts.shape[0]

    if args.VISUALIZE:
        import matplotlib.pyplot as plt
        o3d.visualization.draw_geometries([pcd])

    pts_xid = np.asarray(pts[:,0] // args.cube_size, dtype = np.int32 )
    pts_yid = np.asarray(pts[:,1] // args.cube_size, dtype = np.int32 )
    pts_zid = np.asarray(pts[:,2] // args.cube_size, dtype = np.int32 )
    pts_id = np.vstack([pts_xid, pts_yid, pts_zid]).T

    volume = {}
    for i in tqdm(range(n_pts), desc="Dividing points into cudes", leave=True):
        xid, yid, zid = pts_id[i]
        x, y, z = pts[i]
        r, g, b = colors[i]
        key = f'{xid} {yid} {zid}'
        if key in volume.keys():
            volume[key].append((x,y,z,r,g,b))
        else:
            volume[key] = [(x,y,z,r,g,b)]

    pts_key = volume.keys()
    n_cute = len(pts_key)
    print(f"Average points per cute is {n_pts / n_cute}")
    max_pts_in_cute = 0
    pts_in_cute_list = []
    for key in pts_key:
        n_pts_in_cute = len(volume[key])
        pts_in_cute_list.append(n_pts_in_cute)
        max_pts_in_cute = max(max_pts_in_cute, n_pts_in_cute)
    print(f"Max points in a cute is {max_pts_in_cute}")
    pts_in_cute = np.sort(pts_in_cute_list)

    if args.VISUALIZE:
        plt.plot(pts_in_cute)
        plt.title("Linear graph")
        plt.show()

    cutoff_threshold = pts_in_cute[int(args.cutoff_percent * len(pts_in_cute))]

    key_to_del = []
    for key in pts_key:
        n_pts_in_cute = len(volume[key])
        if n_pts_in_cute < cutoff_threshold:
            key_to_del.append(key)

    for key in key_to_del:
        del volume[key]

    pts_key = volume.keys()
    n_cute = len(pts_key)
    print(f"After cutoff {100 * args.cutoff_percent}% cutes, average points per cute is {n_pts / n_cute}")

    alpha_list = []
    texture_list = []
    position_list = []
    sid_list = []
    with tqdm(total=len(pts_key), desc="Rasterizing points in each cube", leave=True) as pbar:
        for key in pts_key:
            pts_in_cute = np.asarray(volume[key])
            xid, yid, zid = key.split(' ')
            cx = int(xid) * args.cube_size
            cy = int(yid) * args.cube_size
            cz = int(zid) * args.cube_size
            pts_in_cute[:,0] -= cx
            pts_in_cute[:,1] -= cy
            pts_in_cute[:,2] -= cz
            texture, alpha, sid = cute_rasterization(pts_in_cute, args.cube_size, args.texture_size) 
            alpha_list.append(alpha)
            texture_list.append(texture)
            position_list.append(np.array([cx, cy, cz]))
            sid_list.append(sid)
            pbar.update(1)

    alphas = np.asarray(alpha_list)
    textures = np.asarray(texture_list)
    positions = np.asarray(position_list, dtype=np.int32)
    sids = np.asarray(sid_list, dtype=np.uint8)

    utility.write(args, alphas=alphas, textures=textures, positions=positions, sids=sids)