import numpy as np
import os 
import imageio

def write(args, alphas=None, textures=None, positions=None, sids=None, labels=None):
    try:
        os.makedirs(f'data/{args.exp_name}/cube_{args.cube_size}')
    except:
        pass
    if alphas is not None: np.save(f'data/{args.exp_name}/cube_{args.cube_size}/alphas{args.texture_size}', alphas)
    if textures is not None: np.save(f'data/{args.exp_name}/cube_{args.cube_size}/textures{args.texture_size}', textures)
    if positions is not None: np.save(f'data/{args.exp_name}/cube_{args.cube_size}/positions{args.texture_size}', positions)
    if sids is not None: np.save(f'data/{args.exp_name}/cube_{args.cube_size}/sids{args.texture_size}', sids)
    if labels is not None: np.save(f'data/{args.exp_name}/cube_{args.cube_size}/labels{args.texture_size}', labels)

def read(args, keywords=[]):
    return [np.load(f'data/{args.exp_name}/cube_{args.cube_size}/{k}{args.texture_size}.npy') for k in keywords]

def read_texture_lib(args):
    texture_lib = []
    for i in range(args.n_lib):
        im = imageio.imread(f'{args.texture_lib_path}/{i+1}.png')
        im = im.astype(np.float32) / 255.
        if im.shape[-1] > 3:
            imr = im[:,:,0] * (im[:,:,3]).astype(np.float32)
            img = im[:,:,1] * (im[:,:,3]).astype(np.float32)
            imb = im[:,:,2] * (im[:,:,3]).astype(np.float32)
            im = np.stack([imr, img, imb], -1)
        texture_lib.append(im)
    return texture_lib
    
def read_cluster_texture_lib(args):
    texture_lib = []
    for i in range(args.n_lib):
        im = imageio.imread(f'{args.cluster_texture_lib_path}/{i+1}.png')
        im = im.astype(np.float32) / 255.
        if im.shape[-1] > 3:
            imr = im[:,:,0] * (im[:,:,3]).astype(np.float32)
            img = im[:,:,1] * (im[:,:,3]).astype(np.float32)
            imb = im[:,:,2] * (im[:,:,3]).astype(np.float32)
            im = np.stack([imr, img, imb], -1)
        texture_lib.append(im)
    return texture_lib