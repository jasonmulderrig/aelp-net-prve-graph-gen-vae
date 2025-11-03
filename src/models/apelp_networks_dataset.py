import numpy as np
import torch
from torch_geometric.data import (
    InMemoryDataset,
    Data
)
from src.file_io.file_io import (
    root_filepath_str,
    _config_filename_str
)
from src.networks.apelp_networks_config import (
    sample_params_arr_func,
    sample_config_params_arr_func
)
from src.networks.aelp_networks import (
    aelp_multiedge_max,
    nrmlzd_k_func,
    nrmlzd_L_func,
    nrmlzd_l_cntr_func,
    nrmlzd_en_func
)
from src.helpers.graph_utils import (
    lexsorted_edges,
    conn_edges_to_edge_index_arr,
    conn_edges_attr_to_edge_attr_arr
)

def valid_params(cfg):
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    if not np.allclose(sample_config_params_arr[0, 1], sample_config_params_arr[:, 1]):
        error_str = (
            "All apelp networks in the dataset must be of the same "
            + "dimensionality. Please modify the apelp networks "
            + "accordingly."
        )
        raise ValueError(error_str)
    if not np.allclose(sample_config_params_arr[0, 2], sample_config_params_arr[:, 2]):
        error_str = (
            "All apelp networks in the dataset must have the same "
            + "chain segment/cross-linker diameter. Please modify the "
            + "apelp networks accordingly."
        )
        raise ValueError(error_str)
    if not np.allclose(sample_config_params_arr[0, 5], sample_config_params_arr[:, 5]):
        error_str = (
            "All apelp networks in the dataset must have the same "
            + "segment number density. Please modify the apelp "
            + "networks accordingly."
        )
        raise ValueError(error_str)
    if not np.allclose(sample_config_params_arr[0, 6], sample_config_params_arr[:, 6]):
        error_str = (
            "All apelp networks in the dataset must have the same "
            + "maximal cross-linker functionality. Please modify the "
            + "apelp networks accordingly."
        )
        raise ValueError(error_str)
    if not np.allclose(sample_config_params_arr[0, 7], sample_config_params_arr[:, 7]):
        error_str = (
            "All apelp networks in the dataset must have the same "
            + "initial number of core cross-linkers. Please modify the "
            + "apelp networks accordingly."
        )
        raise ValueError(error_str)
    if not np.allclose(sample_config_params_arr[0, 9], sample_config_params_arr[:, 9]):
        error_str = (
            "All apelp networks in the dataset must have the same "
            + "maximum number of segment particles that could compose "
            + "a chain. Please modify the apelp networks accordingly."
        )
        raise ValueError(error_str)
    dim = int(sample_config_params_arr[0, 1])
    b = sample_config_params_arr[0, 2]
    rho_en = sample_config_params_arr[0, 5]
    k = int(sample_config_params_arr[0, 6])
    n = int(sample_config_params_arr[0, 7])
    en_max = int(sample_config_params_arr[0, 9])
    multiedge_max = aelp_multiedge_max(k)

    if dim != 3:
        error_str ="All apelp networks in the dataset must be 3-dimensional."
        raise ValueError(error_str)

    return dim, b, rho_en, k, n, en_max, multiedge_max

def raw_graphs_file_names(cfg):
    date = cfg.networks.apelp.label.date
    batch = cfg.networks.apelp.label.batch
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    raw_file_names = []
    for indx in range(sample_config_num):
        sample = int(sample_config_params_arr[indx, 0])
        config = int(sample_config_params_arr[indx, 10])
        raw_file_names.append(
            _config_filename_str(date, batch, sample, config)+".npz")
    return raw_file_names

def process_graphs(cfg, files):
    # Helper function to load, process, and save specified graph files
    dim, b, _, k, _, en_max, multiedge_max = valid_params(cfg)
    assert dim == 3
    nu_max = en_max + 1
    l_cntr_max = nu_max * b
    graphs = []
    for file in files:
        # Initialize graph data dictionary
        graph_data = {}

        # Load graph data
        npz_graph = np.load(file)

        # Extract data for the as-provided network
        xi = npz_graph["xi"]
        chi = npz_graph["chi"]
        en_mean = npz_graph["en_mean"]
        L = npz_graph["L"]
        L_mean = npz_graph["L_mean"]
        coords = npz_graph["coords"]
        core_nodes_type = npz_graph["core_nodes_type"]
        conn_edges = npz_graph["conn_edges"]
        conn_edges_type = npz_graph["conn_edges_type"]
        l_cntr_conn_edges = npz_graph["l_cntr_conn_edges"]
        lcl_k = npz_graph["lcl_k"]
        lcl_k_diff = npz_graph["lcl_k_diff"]
        lcl_c = npz_graph["lcl_c"]
        lcl_avrg_nn_k = npz_graph["lcl_avrg_nn_k"]
        lcl_lcl_avrg_kappa = npz_graph["lcl_lcl_avrg_kappa"]
        lcl_l = npz_graph["lcl_l"]
        lcl_l_naive = npz_graph["lcl_l_naive"]
        eeel_dobrynin_kappa = npz_graph["eeel_dobrynin_kappa"]
        eeel_glbl_mean_gamma = npz_graph["eeel_glbl_mean_gamma"]
        
        # Number of nodes
        graph_data["num_nodes"] = np.shape(coords)[0]

        # Adjust the core nodes type and edges type labels to start from
        # zero
        core_nodes_type -= 1
        conn_edges_type -= 1

        # Extract the simulation box square/cube side length and its
        # mean value
        if not np.allclose(L[0], L) or not np.allclose(L_mean[0], L_mean):
            error_str = (
                "The simulation box of each apelp network must "
                + "be either a square or a cube. Please modify "
                + "the apelp network simulation box "
                + "accordingly."
            )
            raise ValueError(error_str)
        L = np.asarray([L[0]])
        L_mean = np.asarray([L_mean[0]])

        # Lexicographically sort the edges and edge attributes for the
        # as-provided network
        conn_edges, lexsort_indcs = lexsorted_edges(
            conn_edges, return_indcs=True)
        conn_edges_type = conn_edges_type[lexsort_indcs]
        l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]
        lcl_k_diff = lcl_k_diff[lexsort_indcs]
        lcl_l = lcl_l[lexsort_indcs]
        lcl_l_naive = lcl_l_naive[lexsort_indcs]
        
        # Format the edge index tensor from the edges array for the
        # as-provided network
        graph_data["edge_index"] = (
            torch.from_numpy(conn_edges_to_edge_index_arr(conn_edges)).long()
        )
        
        # Correspondingly modify all edge attributes to match the format
        # of the edge index tensor for the as-provided network
        conn_edges_type = conn_edges_attr_to_edge_attr_arr(conn_edges_type)
        l_cntr_conn_edges = conn_edges_attr_to_edge_attr_arr(l_cntr_conn_edges)
        lcl_k_diff = conn_edges_attr_to_edge_attr_arr(lcl_k_diff)
        lcl_l = conn_edges_attr_to_edge_attr_arr(lcl_l)
        lcl_l_naive = conn_edges_attr_to_edge_attr_arr(lcl_l_naive)

        # Node type for the as-provided network and the multiedge order
        # networks
        graph_data["x_type"] = torch.from_numpy(core_nodes_type).float()
        # Edge type for the as-provided network
        graph_data["edge_type"] = torch.from_numpy(conn_edges_type).float()
        
        # Normalized degree, degree difference, average nearest neighbor
        # degree, and local average nodal connectivity for the
        # as-provided network
        nrmlzd_lcl_k = nrmlzd_k_func(lcl_k, k)
        nrmlzd_lcl_k_diff = nrmlzd_k_func(lcl_k_diff, (k-1))
        nrmlzd_lcl_avrg_nn_k = nrmlzd_k_func(lcl_avrg_nn_k, k)
        nrmlzd_lcl_lcl_avrg_kappa = nrmlzd_k_func(lcl_lcl_avrg_kappa, k)

        # Normalized Euclidean edge length and naive Euclidean edge
        # length for the as-provided network
        nrmlzd_lcl_l = nrmlzd_L_func(lcl_l, L)
        nrmlzd_lcl_l_naive = nrmlzd_L_func(lcl_l_naive, L)

        # Normalized edge contour length, i.e., normalized chain segment
        # number for the as-provided network
        nrmlzd_l_cntr_conn_edges = nrmlzd_l_cntr_func(
            l_cntr_conn_edges, b, l_cntr_max)

        # Normalized mean segment particle number for the as-provided
        # network
        nrmlzd_en_mean = nrmlzd_en_func(en_mean, en_max)

        # Local node attributes for the as-provided network
        x = torch.from_numpy(
            np.column_stack(
                (
                    core_nodes_type, nrmlzd_lcl_k, lcl_c, nrmlzd_lcl_avrg_nn_k,
                    nrmlzd_lcl_lcl_avrg_kappa
                )))
        graph_data["x"] = x.float()
        
        # Local edge attributes for the as-provided network
        edge_attr = torch.from_numpy(
            np.column_stack(
                (
                    conn_edges_type, nrmlzd_l_cntr_conn_edges,
                    nrmlzd_lcl_k_diff, nrmlzd_lcl_l, nrmlzd_lcl_l_naive
                )))
        graph_data["edge_attr"] = edge_attr.float()
        
        # Normalized edge contour length, i.e., normalized chain segment
        # number for the as-provided network
        graph_data["edge_l_cntr"] = (
            torch.from_numpy(nrmlzd_l_cntr_conn_edges).float()
        )

        # Graph-level attributes for the as-provided network
        graph_data["eeel_dobrynin_kappa"] = (
            torch.from_numpy(eeel_dobrynin_kappa).float()
        )
        graph_data["eeel_glbl_mean_gamma"] = (
            torch.from_numpy(eeel_glbl_mean_gamma).float()
        )
        graph_data["xi"] = torch.from_numpy(xi).float()
        graph_data["chi"] = torch.from_numpy(chi).float()
        graph_data["en_mean"] = torch.from_numpy(nrmlzd_en_mean).float()
        
        # Assess each multiedge order network
        for multiedge in range(multiedge_max):
            # Extract data for the multiedge order network
            conn_edges = npz_graph[f"conn_edges_{multiedge:d}"]
            conn_edges_type = npz_graph[f"conn_edges_type_{multiedge:d}"]
            l_cntr_conn_edges = npz_graph[f"l_cntr_conn_edges_{multiedge:d}"]
            lcl_k = npz_graph[f"lcl_k_{multiedge:d}"]
            lcl_k_diff = npz_graph[f"lcl_k_diff_{multiedge:d}"]
            lcl_c = npz_graph[f"lcl_c_{multiedge:d}"]
            lcl_avrg_nn_k = npz_graph[f"lcl_avrg_nn_k_{multiedge:d}"]
            lcl_lcl_avrg_kappa = npz_graph[f"lcl_lcl_avrg_kappa_{multiedge:d}"]
            lcl_l = npz_graph[f"lcl_l_{multiedge:d}"]
            lcl_l_naive = npz_graph[f"lcl_l_naive_{multiedge:d}"]

            # Adjust the edges type labels to start from zero
            conn_edges_type -= 1

            # If the multiedge order network exists, then
            # lexicographically sort its edges and edge attributes
            if not np.array_equal(conn_edges, np.asarray([])):
                conn_edges, lexsort_indcs = lexsorted_edges(
                    conn_edges, return_indcs=True)
                conn_edges_type = conn_edges_type[lexsort_indcs]
                l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]
                lcl_k_diff = lcl_k_diff[lexsort_indcs]
                lcl_l = lcl_l[lexsort_indcs]
                lcl_l_naive = lcl_l_naive[lexsort_indcs]
            
            # Format the edge index tensor from the edges array for the
            # multiedge order network
            graph_data[f"edge_index_{multiedge:d}"] = (
                torch.from_numpy(conn_edges_to_edge_index_arr(conn_edges)).long()
            )
            
            # Correspondingly modify all edge attributes to match the
            # format of the edge index tensor for the multiedge order
            # network
            conn_edges_type = conn_edges_attr_to_edge_attr_arr(conn_edges_type)
            l_cntr_conn_edges = conn_edges_attr_to_edge_attr_arr(
                l_cntr_conn_edges)
            lcl_k_diff = conn_edges_attr_to_edge_attr_arr(lcl_k_diff)
            lcl_l = conn_edges_attr_to_edge_attr_arr(lcl_l)
            lcl_l_naive = conn_edges_attr_to_edge_attr_arr(lcl_l_naive)
            
            # Edge type for the multiedge order network
            graph_data[f"edge_type_{multiedge:d}"] = (
                torch.from_numpy(conn_edges_type).long()
            )

            # Normalized degree, average nearest neighbor degree, and
            # local average nodal connectivity for the multiedge order
            # network
            nrmlzd_lcl_k = nrmlzd_k_func(lcl_k, 2)
            nrmlzd_lcl_avrg_nn_k = nrmlzd_k_func(lcl_avrg_nn_k, 2)
            nrmlzd_lcl_lcl_avrg_kappa = nrmlzd_k_func(lcl_lcl_avrg_kappa, 2)

            # Normalized Euclidean edge length and naive Euclidean edge
            # length for the multiedge order network
            nrmlzd_lcl_l = nrmlzd_L_func(lcl_l, L)
            nrmlzd_lcl_l_naive = nrmlzd_L_func(lcl_l_naive, L)

            # Normalized edge contour length, i.e., normalized chain
            # segment number for the multiedge order network
            nrmlzd_l_cntr_conn_edges = nrmlzd_l_cntr_func(
                l_cntr_conn_edges, b, l_cntr_max)

            # Normalized mean segment particle number for the multiedge
            # order network
            nrmlzd_en_mean = nrmlzd_en_func(en_mean, en_max)

            # Local node attributes for the multiedge order network
            x = torch.from_numpy(
                np.column_stack(
                    (
                        core_nodes_type, nrmlzd_lcl_k, lcl_c,
                        nrmlzd_lcl_avrg_nn_k, nrmlzd_lcl_lcl_avrg_kappa
                    )))
            graph_data[f"x_{multiedge:d}"] = x.float()
            
            # Local edge attributes for the multiedge order network
            edge_attr = torch.from_numpy(
                np.column_stack(
                    (
                        conn_edges_type, nrmlzd_l_cntr_conn_edges,
                        lcl_k_diff, nrmlzd_lcl_l, nrmlzd_lcl_l_naive
                    )))
            graph_data[f"edge_attr_{multiedge:d}"] = edge_attr.float()

            # Normalized edge contour length for the multiedge order
            # network, i.e., normalized chain segment number for the
            # multiedge order network
            graph_data[f"edge_l_cntr_{multiedge:d}"] = (
                torch.from_numpy(nrmlzd_l_cntr_conn_edges).float()
            )
        
        # Normalize nodal coordinates
        nrmlzd_coords = nrmlzd_L_func(coords, L)
        graph_data["coords"] = torch.from_numpy(nrmlzd_coords).float()
        
        # Mean simulation box side length and normalized simulation box
        # side length 
        nrmlzd_L = nrmlzd_L_func(L, L_mean)
        graph_data["L"] = torch.from_numpy(nrmlzd_L).float()
        graph_data["L_mean"] = torch.from_numpy(L_mean).float()

        # Instantiate a PyTorch Geometric graph, and add the graph to
        # the split graph list
        graphs.append(Data(**graph_data))
    return graphs

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
        self.cfg = cfg
        self.split = split
        root = root_filepath_str(self.cfg.networks.apelp.label.network)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return raw_graphs_file_names(self.cfg)
    
    @property
    def processed_file_names(self):
        date = self.cfg.networks.apelp.label.date
        batch = self.cfg.networks.apelp.label.batch
        if self.split == "train": return [f"{date}{batch}-train.pt"]
        elif self.split == "val": return [f"{date}{batch}-val.pt"]
        else: return [f"{date}{batch}-test.pt"]
    
    def download(self): pass

    def process(self):
        # Gather sample and configuration numbers
        _, sample_num = sample_params_arr_func(self.cfg)
        config_num = self.cfg.networks.apelp.topology.config

        # Initialize random number generator
        rng = np.random.default_rng(self.cfg.general.seed)
        
        # Shuffle and split configurations for stratified splitting
        perm_configs = rng.permutation(config_num)
        num_train_configs = int(self.cfg.general.train_split*config_num)
        num_val_configs = int(self.cfg.general.val_split*config_num)
        train_configs = perm_configs[:num_train_configs]
        val_configs = (
            perm_configs[num_train_configs:(num_train_configs+num_val_configs)]
        )
        test_configs = perm_configs[(num_train_configs+num_val_configs):]
        
        # Gather train, validation, and test files via the stratified
        # splitting method (while also randomizing sample order)
        train_files = []
        for sample in rng.permutation(sample_num):
            for config in train_configs:
                train_files.append(self.raw_paths[(sample*config_num)+config])
        
        val_files = []
        for sample in rng.permutation(sample_num):
            for config in val_configs:
                val_files.append(self.raw_paths[(sample*config_num)+config])
        
        test_files = []
        for sample in rng.permutation(sample_num):
            for config in test_configs:
                test_files.append(self.raw_paths[(sample*config_num)+config])

        # Process and save the split data
        if self.split == "train":
            self.save(
                process_graphs(self.cfg, train_files), self.processed_paths[0])
        elif self.split == "val":
            self.save(
                process_graphs(self.cfg, val_files), self.processed_paths[0])
        else:
            self.save(
                process_graphs(self.cfg, test_files), self.processed_paths[0])
