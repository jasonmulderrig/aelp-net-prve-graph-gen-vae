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
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

class Namespace(): pass

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    aelp_network = list(cfg.networks.keys())[0]
    if aelp_network != "apelp":
        print("The specified network must be ``apelp''.")
        return None
    
    # Extract latent space dimensions for the attributed network
    # embedding method
    dim, _, _, _, _, _, multiedge_max = valid_params(cfg)
    obj = Namespace()
    obj = aelp_d_func(
        obj, cfg.model.d_adj, cfg.model.d_coords, cfg.model.d_adj_coords,
        multiedge_max)
    d = obj.d

    # Gather dataset
    train_dataset = apelpDataset(cfg, "train")
    val_dataset = apelpDataset(cfg, "val")
    test_dataset = apelpDataset(cfg, "test")

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Determine the number of nodes in each graph in the dataset
    data = next(iter(train_dataloader))
    batch, n, batch_size = batch_analysis(data)
    n_graph = n // batch_size
    assert (n%batch_size) == 0

    # Determine the total number of graphs in the dataset
    num_sample_configs = (
        len(train_dataloader) + len(val_dataloader) + len(test_dataloader)
    )

    # Load the trained model
    model = apelpAVGAE(cfg, n_graph).to(device)
    for params in model.parameters():
        if params.dim() == 1: nn.init.constant_(params, 0)
        else: nn.init.xavier_normal_(params)
    chkpnt = torch.load(
        chkpnt_filename_str(aelp_network, cfg.label)+".model", weights_only=False)
    model.load_state_dict(chkpnt["model_state_dict"])
    
    # Initialize tensors to store mu and logstd that the trained model
    # calculates for each graph in the dataset
    sample_configs_mu = torch.empty(
        (num_sample_configs, n_graph, d), dtype=torch.float)
    sample_configs_logstd = torch.empty(
        (num_sample_configs, n_graph, d), dtype=torch.float)
    
    indx = 0
    model.eval()
    with torch.no_grad():
        for data in train_dataloader:
            # Extract batched data to the GPU
            data = data.to(device)

            mu, logstd = model.encoder(data)
            sample_configs_mu[indx] = mu
            sample_configs_logstd[indx] = logstd
            indx += 1
        for data in val_dataloader:
            # Extract batched data to the GPU
            data = data.to(device)

            mu, logstd = model.encoder(data)
            sample_configs_mu[indx] = mu
            sample_configs_logstd[indx] = logstd
            indx += 1
        for data in test_dataloader:
            # Extract batched data to the GPU
            data = data.to(device)

            mu, logstd = model.encoder(data)
            sample_configs_mu[indx] = mu
            sample_configs_logstd[indx] = logstd
            indx += 1
    
    sample_configs_mu = sample_configs_mu.detach().cpu().numpy()
    sample_configs_logstd = sample_configs_logstd.detach().cpu().numpy()

    filepath = filepath_str("analysis")
    sample_configs_mu_filename = filepath + "sample_configs_mu" + ".npy"
    sample_configs_logstd_filename = filepath + "sample_configs_logstd" + ".npy"

    np.save(sample_configs_mu_filename, sample_configs_mu)
    np.save(sample_configs_logstd_filename, sample_configs_logstd)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Trained latent space parameter calculation took {execution_time} seconds to run")