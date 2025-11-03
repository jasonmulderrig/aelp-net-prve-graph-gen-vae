# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.models.apelp_networks_dataset import (
    valid_params,
    apelpDataset
)
from src.models.apelp_networks_avgae import apelpAVGAE
from src.helpers.model_utils import (
    aelp_d_func,
    batch_analysis
)
from src.file_io.file_io import (
    chkpnt_filename_str,
    filepath_str
)
from src.networks.aelp_networks import (
    aelp_multiedge_max,
    dmnsnlzd_k_func,
    dmnsnlzd_L_func,
    dmnsnlzd_l_cntr_func,
    dmnsnlzd_en_func
)
from src.helpers.graph_utils import (
    lexsorted_edges,
    conn_edges_to_edge_index_arr,
    edge_index_arr_to_conn_edges,
    conn_edges_attr_to_edge_attr_arr,
    edge_attr_arr_to_conn_edges_attr
)
from src.descriptors.general_topological_descriptors import l_func
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    aelp_network = list(cfg.networks.keys())[0]
    if aelp_network != "apelp":
        print("The specified network must be ``apelp''.")
        return None
    assert cfg.model.synthesis_protocol_params == True
    assert cfg.model.property_descriptors == True
    
    # Load in the trained latent space parameters
    filepath = filepath_str("analysis")
    sample_configs_mu_filename = filepath + "sample_configs_mu" + ".npy"
    sample_configs_mu = np.load(sample_configs_mu_filename)
    num_sample_configs, n_graph, d = np.shape(sample_configs_mu)
    # # Take the mean of the latent space means as the sampled latent
    # # space
    # z = np.mean(sample_configs_mu, axis=0)
    # z = torch.from_numpy(z).float()
    # Take the mean of the latent space from the first network
    z = torch.from_numpy(sample_configs_mu[0]).float()
    
    # Gather fundamental dataset-wide parameters
    dim, b, _, k, _, en_max, multiedge_max = valid_params(cfg)
    assert dim == 3
    nu_max = en_max + 1
    l_cntr_max = nu_max * b
    data = next(iter(DataLoader(apelpDataset(cfg, "train"), batch_size=1, shuffle=False)))
    batch, n, batch_size = batch_analysis(data)
    n_graph = n // batch_size
    assert (n%batch_size) == 0
    L_mean = data.L_mean.item()

    # Load the trained model
    model = apelpAVGAE(cfg, n_graph).to(device)
    for params in model.parameters():
        if params.dim() == 1: nn.init.constant_(params, 0)
        else: nn.init.xavier_normal_(params)
    chkpnt = torch.load(
        chkpnt_filename_str(aelp_network, cfg.label)+".model", weights_only=False)
    model.load_state_dict(chkpnt["model_state_dict"])

    # Decode the sampled latent space into its network graph topology,
    # synthesis parameters, and network topology parameters
    model.eval()
    with torch.no_grad():
        (xi, chi, en_mean, L, coords, edge_index, edge_type, edge_l_cntr,
         eeel_dobrynin_kappa, eeel_glbl_mean_gamma) = model.decoder.forward_all(z)
    
    # Convert Torch tensors to Numpy arrays, properly structure the
    # simulation box side lengths array, and convert edge type entries
    # to integers
    xi = np.asarray([xi.item()])
    chi = np.asarray([chi.item()])
    en_mean = np.asarray([en_mean.item()])
    L_mean = np.asarray([L_mean])
    L = np.repeat(L.item(), dim)
    coords = coords.detach().cpu().numpy()
    edge_index = edge_index.detach().cpu().numpy()
    edge_type = edge_type.long().detach().cpu().numpy()
    edge_l_cntr = edge_l_cntr.detach().cpu().numpy()
    eeel_dobrynin_kappa = np.asarray([eeel_dobrynin_kappa.item()])
    eeel_glbl_mean_gamma = np.asarray([eeel_glbl_mean_gamma.item()])

    # Convert edge index tensor to edges array, and correspondingly
    # modify all edge attributes to match the format of the edges array
    conn_edges = edge_index_arr_to_conn_edges(edge_index)
    conn_edges_type = edge_attr_arr_to_conn_edges_attr(edge_type)
    l_cntr_conn_edges = edge_attr_arr_to_conn_edges_attr(edge_l_cntr)

    # Readjust the edges type labels
    conn_edges_type += 1

    # Lexicographically sort the edges and edge attributes
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]
    l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]

    # Acquire number of decoded edges
    m_decoded = np.shape(conn_edges)[0]
    
    # Dimensionalize decoded parameters that are supplied
    # nondimensionally
    print(en_mean)
    print(l_cntr_conn_edges)
    print(L)
    print(coords)
    print("\n")
    en_mean = dmnsnlzd_en_func(en_mean, en_max)
    l_cntr_conn_edges = dmnsnlzd_l_cntr_func(l_cntr_conn_edges, b, l_cntr_max)
    L = dmnsnlzd_L_func(L, L_mean)
    coords = dmnsnlzd_L_func(coords, L)
    print(en_mean)
    print(l_cntr_conn_edges)
    print(L)
    print(coords)
    print("\n")
    print(conn_edges_type)

    # Calculate Euclidean edge lengths
    l = l_func(conn_edges, conn_edges_type, coords, L)
    print(l)

    # Gather edges whose Euclidean edge length is less than or equal to
    # their contour length
    l_leq_l_cntr_indcs = np.where(l <= l_cntr_conn_edges)[0]
    print(m_decoded, l_leq_l_cntr_indcs)







if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Trained latent space graph generation and re-encoding took {execution_time} seconds to run")