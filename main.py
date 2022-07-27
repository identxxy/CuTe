from cute_voxelize import cute_voxelize
from cute_cluster import cute_cluster
from cute_render import cute_render

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
        help='config file path')
    parser.add_argument("--exp_name", type=str, default='unnamed',
        help='give a name to the model')
    parser.add_argument("--pcd_filename", type=str, 
        help='pointcloud filename')
    parser.add_argument("--cube_size", type=float, default=1.0, 
        help='voxelization cube size (meter)')
    parser.add_argument("--texture_size", type=int, default=32, 
        help='cube texture size (int)')
    parser.add_argument("--cutoff_percent", type=float, default=0.3, 
        help='percentage of cubes with few points insides will be deleted.')
    parser.add_argument("--feature_bins", type=int, default=8, 
        help='number of bins of texture feature histogram.')
    # TEXTURE_LIB
    parser.add_argument('--USE_TEXTURE_LIB', action='store_true',
        help='whether to use provided texture libs.')
    parser.add_argument('--USE_CLUSTER_TEXTURE_LIB', action='store_true',
        help='whether to use provided texture libs to initialize clustering.')
    parser.add_argument("--texture_lib_path", type=str, default='textures', 
        help='path to textures lib dir.')
    parser.add_argument("--cluster_texture_lib_path", type=str, default='textures/cluster', 
        help='path to textures lib dir that are used for clustering.')
    parser.add_argument("--n_lib", type=int, default=8, 
        help='number of texture lib samples.')
    parser.add_argument("--n_textures", type=int, default=8,
        help='cluster target number of textures, if not USE_TEXTURE_LIB')
    # module control
    parser.add_argument('--RAST_ONLY', action='store_true',
        help='only do rasterization.')
    parser.add_argument('--CLUSTER_ONLY', action='store_true',
        help='only do clustering.')
    parser.add_argument('--RENDER_ONLY', action='store_true',
        help='only do rendering.')
    parser.add_argument('--RENDER_RAST', action='store_true',
        help='render the original texture (rasterization result).')

    parser.add_argument('--VISUALIZE', action='store_true',
                        help='whether to plot intermediate results.')
    parser.add_argument("--n_texture_eg_to_show", type=int, default=8,
        help='number of textures per label to show.')
    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    if (not args.CLUSTER_ONLY) and (not args.RENDER_ONLY):
        cute_voxelize(args)
    if (not args.RAST_ONLY) and (not args.RENDER_ONLY):
        cute_cluster(args)
    if (not args.RAST_ONLY) and (not args.CLUSTER_ONLY):
        cute_render(args)
    
    