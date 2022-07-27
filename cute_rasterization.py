import numpy as np

# pts_in_cute: cx, cy, cz, r, g, b. cx/y/z is the coord inside the cute.
def cute_rasterization(pts_in_cute, cube_size, texture_size):
    dist = cube_size / texture_size
    pts_in_cute[:,0] /= dist
    pts_in_cute[:,1] /= dist
    pts_in_cute[:,2] /= dist
    xyz_max = pts_in_cute[:,0:3].max(0)
    xyz_min = pts_in_cute[:,0:3].min(0)
    xyz_mean = pts_in_cute[:,0:3].mean(0)
    xyz_range = np.max([xyz_max - xyz_mean, xyz_mean - xyz_min], 0)
    n_pts = pts_in_cute.shape[0]
    
    pts_coord_int = pts_in_cute[:, 0:3].astype(np.int32)
    pts_coord_frac = pts_in_cute[:, 0:3] - pts_coord_int - 0.5
    pts_coord_go_right = pts_coord_frac > 0
    # if pts_coord_frac > 0, (frac) is the weight of int, (1-frac) is the weight of int+1
    # if pts_coord_frac < 0, (1+frac) is the weight of int-1, (-frac) is the weight of int
    pts_coord_start = (pts_coord_int - 1 + pts_coord_go_right) % texture_size
    pts_coord_start_frac = pts_coord_go_right * pts_coord_frac + (1-pts_coord_go_right) * (1+pts_coord_frac)
    pts_coord_end = (pts_coord_start + 1) % texture_size
    pts_coord_end_frac = 1 - pts_coord_start_frac
        
    alpha = np.zeros((3, texture_size, texture_size))     # x, y, z
    per_pt_w = np.zeros((n_pts, 12))
    for i in range(n_pts):
        px, py, pz, = pts_in_cute[i][0:3]
        ix1, iy1, iz1 = pts_coord_start[i]
        ix2, iy2, iz2 = pts_coord_end[i]
        ix1w, iy1w, iz1w = pts_coord_start_frac[i]
        ix2w, iy2w, iz2w = pts_coord_end_frac[i]
        
        ### VERY IMPORTANT!
        xw = np.exp(- abs(px - xyz_mean[0]) / xyz_range[0])
        yw = np.exp(- abs(py - xyz_mean[1]) / xyz_range[1])
        zw = np.exp(- abs(pz - xyz_mean[2]) / xyz_range[2])
        # project onto x
        per_pt_w[i, 0] = (iy1w * iz1w) * xw
        per_pt_w[i, 1] = (iy1w * iz2w) * xw
        per_pt_w[i, 2] = (iy2w * iz1w) * xw 
        per_pt_w[i, 3] = (iy2w * iz2w) * xw       
        alpha[0, iy1, iz1] += per_pt_w[i, 0] 
        alpha[0, iy1, iz2] += per_pt_w[i, 1]
        alpha[0, iy2, iz1] += per_pt_w[i, 2]
        alpha[0, iy2, iz2] += per_pt_w[i, 3]
        # project onto y
        per_pt_w[i, 4] = (ix1w * iz1w) * yw
        per_pt_w[i, 5] = (ix1w * iz2w) * yw 
        per_pt_w[i, 6] = (ix2w * iz1w) * yw 
        per_pt_w[i, 7] = (ix2w * iz2w) * yw
        alpha[1, ix1, iz1] += per_pt_w[i, 4]
        alpha[1, ix1, iz2] += per_pt_w[i, 5]
        alpha[1, ix2, iz1] += per_pt_w[i, 6]
        alpha[1, ix2, iz2] += per_pt_w[i, 7]
        # project onto z
        per_pt_w[i, 8] += (ix1w * iy1w) * zw 
        per_pt_w[i, 9] += (ix1w * iy2w) * zw 
        per_pt_w[i, 10] += (ix2w * iy1w) * zw 
        per_pt_w[i, 11] += (ix2w * iy2w) * zw
        alpha[2, ix1, iy1] += per_pt_w[i, 8]
        alpha[2, ix1, iy2] += per_pt_w[i, 9]
        alpha[2, ix2, iy1] += per_pt_w[i, 10]
        alpha[2, ix2, iy2] += per_pt_w[i, 11]
        
    alpha_sum = np.sum(alpha, axis=(1,2))
    amax = np.argmax(alpha_sum)
    rst_alpha = alpha[amax]
    rst_texture = np.zeros((3, texture_size, texture_size)) # r, g, b
    for i in range(n_pts):
        rgb = pts_in_cute[i][3:6]
        w1 = per_pt_w[i, 4*amax+0]
        w2 = per_pt_w[i, 4*amax+1]
        w3 = per_pt_w[i, 4*amax+2]
        w4 = per_pt_w[i, 4*amax+3]
        
        u_axis = (amax + 1) % 3
        v_axis = (amax + 2) % 3
        u1 = pts_coord_start[i, u_axis]
        u2 = pts_coord_end[i, u_axis]
        v1 = pts_coord_start[i, v_axis]
        v2 = pts_coord_end[i, v_axis]
        
        for i in range(3):
            rst_texture[i, u1, v1] += rgb[i] * w1
            rst_texture[i, u1, v2] += rgb[i] * w2
            rst_texture[i, u2, v1] += rgb[i] * w3
            rst_texture[i, u2, v2] += rgb[i] * w4
    rst_texture /= n_pts / texture_size**2
    return rst_texture, rst_alpha, amax
