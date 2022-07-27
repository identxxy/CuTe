import numpy as np
from sklearn.cluster import KMeans

import utility 

def feature_extract(texture, bins):
    r = texture[0,...]
    g = texture[1,...]
    b = texture[2,...]
    # r /= max(r.max(), 1.0)
    # g /= max(g.max(), 1.0)
    # b /= max(b.max(), 1.0)
    r[r>1.] = 1.
    g[g>1.] = 1.
    b[b>1.] = 1.
    rhist = np.histogram(r, bins, range=(0,1), density=True)[0]
    ghist = np.histogram(g, bins, range=(0,1), density=True)[0]
    bhist = np.histogram(b, bins, range=(0,1), density=True)[0]
    feature = np.concatenate([rhist, ghist, bhist])
    return feature

def texture_cluster(textures, n_clusters, feature_bins=8, lib=None):
    n_texture = textures.shape[0]
    feature_mat = np.zeros((n_texture, 3 * (feature_bins)))
    for i in range(n_texture):
        feature_mat[i] = feature_extract(textures[i], feature_bins)
    
    if lib:
        n_lib = len(lib)
        feature_lib = np.zeros((n_lib, 3 * (feature_bins)))
        for i in range(n_lib):
            feature_lib[i] = feature_extract(lib[i].T, feature_bins)
        kmeans = KMeans(n_clusters=n_clusters, init=feature_lib)

    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(feature_mat)
    return kmeans.labels_

def cute_cluster(args):
    textures, positions = utility.read(args, ['textures', 'positions'])
    if args.USE_TEXTURE_LIB:
        if args.USE_CLUSTER_TEXTURE_LIB:
            texture_lib = utility.read_cluster_texture_lib(args)
        else:
            texture_lib = utility.read_texture_lib(args)
        labels = texture_cluster(textures, args.n_lib, args.feature_bins, lib=texture_lib)
    else:
        labels = texture_cluster(textures, args.n_textures, args.feature_bins)

    if args.VISUALIZE:
        import matplotlib.pyplot as plt
        for i in range(args.n_lib):
            idx = np.where(labels == i)[0]
            rand_idx = np.random.choice(idx, args.n_texture_eg_to_show)
            eg_textures = textures[rand_idx]
            eg_textures = np.hstack(eg_textures)
            plt.imshow(eg_textures.T)
            plt.title(f'example texture of class {i+1}')
            plt.show()
    
    utility.write(args, labels=labels)

