import numpy as np
import torch
from torch_geometric.data import (
    InMemoryDataset,
    Data
)
from file_io.file_io import (
    root_filepath_str,
    _config_filename_str
)
from networks.apelp_networks_config import sample_config_params_arr_func
from helpers.simulation_box_utils import L_diag_max_func
from helpers.graph_utils import (
    lexsorted_edges,
    unique_conn_edges_and_conn_edges_attr,
    conn_edges_attr_to_edge_index_attr_arr,
    conn_edges_to_edge_index_arr
)

class apelpDataset(InMemoryDataset):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
            self,
            cfg,
            split,
            transform=None,
            pre_transform=None,
            pre_filter=None):
        root = root_filepath_str(cfg.networks.apelp.label.network)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.cfg = cfg
        self.split = split
        self.edge_dim = 5 ### may need to be modified...
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        sample_config_params_arr, sample_config_num = (
            sample_config_params_arr_func(self.cfg)
        )
        raw_file_names_list = []

        date = self.cfg.networks.apelp.label.date
        batch = self.cfg.networks.apelp.label.batch
        for indx in range(sample_config_num):
            sample = int(sample_config_params_arr[indx, 0])
            config = int(sample_config_params_arr[indx, 9])
            raw_file_names_list.append(
                _config_filename_str(date, batch, sample, config)+".npz")
        
        return raw_file_names_list
    
    @property
    def processed_file_names(self):
        if self.split == "train": return ["train.pt"]
        elif self.split == "val": return ["val.pt"]
        else: return ["test.pt"]
    
    def download(self): pass

    def process(self):
        # Gather raw graph files
        raw_files = self.raw_paths
        raw_files = np.asarray(raw_files)
        
        # Shuffle and split raw graph files
        rng = np.random.default_rng(self.cfg.general.seed)
        num_raw_files = len(self.raw_paths)
        perm = rng.permutation(num_raw_files)

        num_train = int(self.cfg.general.train_split*num_raw_files)
        num_val = int(self.cfg.general.val_split*num_raw_files)

        train_files = raw_files[perm[:num_train]]
        val_files = raw_files[perm[num_train:num_train+num_val]]
        test_files = raw_files[perm[num_train+num_val:]]

        # TO-DO:
        # (1) Figure out if I should include lcl_c and conn_edges_counts
        # in the node and edge attribute tensors, respectively, or omit
        # them -- I need to make a coalesced version of conn_edges (clscd_conn_edges) and perform ML on that...

        # Helper function to load, process, and save split graph files
        def process_split(split_files):
            graph_split = []
            for path in split_files:
                # Load graph data
                npz_graph = np.load(path)
                
                L = npz_graph["L"]
                coords = npz_graph["coords"]
                core_nodes_type = npz_graph["core_nodes_type"]
                conn_edges = npz_graph["conn_edges"]
                conn_edges_type = npz_graph["conn_edges_type"]
                lcl_k = npz_graph["lcl_k"]
                lcl_k_diff = npz_graph["lcl_k_diff"]
                lcl_avrg_nn_k = npz_graph["lcl_avrg_nn_k"]
                lcl_c = npz_graph["lcl_c"]
                lcl_lcl_avrg_kappa = npz_graph["lcl_lcl_avrg_kappa"]
                lcl_l = npz_graph["lcl_l"]
                lcl_l_naive = npz_graph["lcl_l_naive"]
                eeel_glbl_mean_k = npz_graph["eeel_glbl_mean_k"]
                glbl_prop_eeel_n = npz_graph["glbl_prop_eeel_n"]
                glbl_prop_eeel_m = npz_graph["glbl_prop_eeel_m"]
                eeel_glbl_mean_gamma = npz_graph["eeel_glbl_mean_gamma"]
                glbl_n_fractal_dim = npz_graph["glbl_n_fractal_dim"]
                glbl_xi_corr = npz_graph["glbl_xi_corr"]
                
                # Number of nodes
                num_nodes = np.shape(core_nodes_type)[0]

                # Lexicographically sort the edges and edge attributes
                conn_edges, lexsort_indcs = lexsorted_edges(
                    conn_edges, return_indcs=True)
                conn_edges_type = conn_edges_type[lexsort_indcs]
                lcl_k_diff = lcl_k_diff[lexsort_indcs]
                lcl_l = lcl_l[lexsort_indcs]
                lcl_l_naive = lcl_l_naive[lexsort_indcs]

                # Unique conn_edges and edge attributes
                _, _, conn_edges_type = unique_conn_edges_and_conn_edges_attr(
                    conn_edges, conn_edges_type)
                _, _, lcl_k_diff = unique_conn_edges_and_conn_edges_attr(
                    conn_edges, lcl_k_diff)
                _, _, lcl_l = unique_conn_edges_and_conn_edges_attr(
                    conn_edges, lcl_l)
                conn_edges, conn_edges_counts, lcl_l_naive = (
                    unique_conn_edges_and_conn_edges_attr(
                        conn_edges, lcl_l_naive)
                )
                
                # Format the edge index tensor from the edges array
                edge_index = torch.from_numpy(
                    conn_edges_to_edge_index_arr(conn_edges))
                
                # Correspondingly modify all edge attributes to match
                # the format of the edge index tensor
                conn_edges_type = conn_edges_attr_to_edge_index_attr_arr(
                    conn_edges_type)
                conn_edges_counts = conn_edges_attr_to_edge_index_attr_arr(
                    conn_edges_counts)
                lcl_k_diff = conn_edges_attr_to_edge_index_attr_arr(lcl_k_diff)
                lcl_l = conn_edges_attr_to_edge_index_attr_arr(lcl_l)
                lcl_l_naive = conn_edges_attr_to_edge_index_attr_arr(lcl_l_naive)

                # Node type, edge type, and edge counts
                x_type = torch.from_numpy(core_nodes_type)
                edge_type = torch.from_numpy(conn_edges_type)
                edge_counts = torch.from_numpy(conn_edges_counts)

                # Normalize nodal coordinates, if called for. Then,
                # universally save the nodal coordinates into a torch
                # tensor.
                if self.cfg.general.normalize_nodal_coordinates:
                    nrmlzd_coords = coords / L
                    pos = torch.from_numpy(nrmlzd_coords)
                else: pos = torch.from_numpy(coords)
                
                # Normalize graph descriptors, if called for, in a
                # physically-meaningful manner. Then, universally save
                # node type, edge type, edge contour length, local node
                # attributes, local edge attributes, and graph-level
                # attributes.
                if self.cfg.general.normalize_graph_descriptors:
                    # Normalized degree, degree difference, average
                    # nearest neighbor degree, and local average nodal
                    # connectivity
                    nrmlzd_lcl_k = lcl_k / self.cfg.networks.apelp.topology.k
                    nrmlzd_lcl_k_diff = (
                        lcl_k_diff / (self.cfg.networks.apelp.topology.k-1)
                    )
                    nrmlzd_lcl_avrg_nn_k = (
                        lcl_avrg_nn_k / self.cfg.networks.apelp.topology.k
                    )
                    nrmlzd_lcl_lcl_avrg_kappa = (
                        lcl_lcl_avrg_kappa / self.cfg.networks.apelp.topology.k
                    )

                    # Normalized Euclidean edge length and naive
                    # Euclidean edge length
                    L_diag_max = L_diag_max_func(L)
                    nrmlzd_lcl_l = lcl_l / L_diag_max
                    nrmlzd_lcl_l_naive = lcl_l_naive / L_diag_max

                    # Normalized global mean degree
                    nrmlzd_eeel_glbl_mean_k = (
                        eeel_glbl_mean_k / self.cfg.networks.apelp.topology.k
                    )

                    # Normlized node-based fractal dimension
                    nrmlzd_glbl_n_fractal_dim = (
                        glbl_n_fractal_dim
                        / self.cfg.networks.apelp.topology.dim
                    )

                    # Normalized correlation length
                    nrmlzd_glbl_xi_corr = glbl_xi_corr / L_diag_max

                    # Local node attributes
                    x = torch.from_numpy(
                        np.column_stack(
                            (
                                core_nodes_type, nrmlzd_lcl_k,
                                nrmlzd_lcl_avrg_nn_k, lcl_c,
                                nrmlzd_lcl_lcl_avrg_kappa)))
                    
                    # Local edge attributes
                    edge_attr = torch.from_numpy(
                        np.column_stack(
                            (
                                conn_edges_type, conn_edges_counts,
                                nrmlzd_lcl_k_diff, nrmlzd_lcl_l,
                                nrmlzd_lcl_l_naive)))

                    # Graph-level attributes
                    eeel_glbl_mean_k = torch.from_numpy(nrmlzd_eeel_glbl_mean_k)
                    glbl_prop_eeel_n = torch.from_numpy(glbl_prop_eeel_n)
                    glbl_prop_eeel_m = torch.from_numpy(glbl_prop_eeel_m)
                    eeel_glbl_mean_gamma = torch.from_numpy(
                        eeel_glbl_mean_gamma)
                    glbl_n_fractal_dim = torch.from_numpy(
                        nrmlzd_glbl_n_fractal_dim)
                    glbl_xi_corr = torch.from_numpy(nrmlzd_glbl_xi_corr)
                else:
                    # Local node attributes
                    x = torch.from_numpy(
                        np.column_stack(
                            (
                                core_nodes_type, lcl_k, lcl_avrg_nn_k, lcl_c,
                                lcl_lcl_avrg_kappa)))
                    
                    # Local edge attributes
                    edge_attr = torch.from_numpy(
                        np.column_stack(
                            (
                                conn_edges_type, conn_edges_counts, lcl_k_diff,
                                lcl_l, lcl_l_naive)))

                    # Graph-level attributes
                    eeel_glbl_mean_k = torch.from_numpy(eeel_glbl_mean_k)
                    glbl_prop_eeel_n = torch.from_numpy(glbl_prop_eeel_n)
                    glbl_prop_eeel_m = torch.from_numpy(glbl_prop_eeel_m)
                    eeel_glbl_mean_gamma = torch.from_numpy(
                        eeel_glbl_mean_gamma)
                    glbl_n_fractal_dim = torch.from_numpy(glbl_n_fractal_dim)
                    glbl_xi_corr = torch.from_numpy(glbl_xi_corr)

                # Instantiate a PyTorch Geometric graph, and add the
                # graph to the split graph list
                graph = Data(
                    num_nodes=num_nodes, pos=pos, x=x, x_type=x_type,
                    edge_index=edge_index, edge_attr=edge_attr,
                    edge_type=edge_type, edge_counts=edge_counts,
                    eeel_glbl_mean_k=eeel_glbl_mean_k,
                    glbl_prop_eeel_n=glbl_prop_eeel_n,
                    glbl_prop_eeel_m=glbl_prop_eeel_m,
                    eeel_glbl_mean_gamma=eeel_glbl_mean_gamma,
                    glbl_n_fractal_dim=glbl_n_fractal_dim,
                    glbl_xi_corr=glbl_xi_corr)
                graph_split.append(graph)
            return graph_split
        
        # Process and save the split data (and slices)
        if self.split == "train":
            self.save(process_split(train_files), self.processed_paths[0])
        elif self.split == "val":
            self.save(process_split(val_files), self.processed_paths[0])
        else:
            self.save(process_split(test_files), self.processed_paths[0])

# n = np.shape(core_nodes_type)[0]
# m = np.shape(conn_edges)[0]

# # Amend the graph such that core edges reside on nodes 0
# # to n-1, and periodic boundary edges reside on nodes n
# # to 2*n-1
# for edge in range(m):
#     conn_edges[edge, 0] = int(
#         (1-conn_edges_type[edge, 0])*n+conn_edges[edge, 0])
#     conn_edges[edge, 1] = int(
#         (1-conn_edges_type[edge, 1])*n+conn_edges[edge, 1])

# # Duplicate the nodal coordinates and node attributes to
# # account for the amended graph structure
# # differentiating core and periodic boundary edges
# coords = np.vstack((coords, coords))
# core_nodes_type = np.concatenate(
#     (core_nodes_type, core_nodes_type), dtype=int)
# lcl_k = np.concatenate((lcl_k, lcl_k), dtype=int)
# lcl_avrg_nn_k = np.concatenate((lcl_avrg_nn_k, lcl_avrg_nn_k))
# lcl_c = np.concatenate((lcl_c, lcl_c))
# lcl_lcl_avrg_kappa = np.concatenate(
#     (lcl_lcl_avrg_kappa, lcl_lcl_avrg_kappa))