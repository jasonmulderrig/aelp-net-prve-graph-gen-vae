# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# from torch_geometric.utils import batched_negative_sampling
from src.networks.apelp_networks_config import sample_config_params_arr_func

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    
    sample = 13
    config = 17

    indx = np.where(np.logical_and(sample_config_params_arr[:, 0]==sample, sample_config_params_arr[:, 10]==config))[0][0]
    assert indx == sample*cfg.networks.apelp.topology.config + config
    print(indx)
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # aelp_network = list(cfg.networks.keys())[0]
    # if aelp_network not in ["abelp", "apelp", "auelp"]:
    #     error_str = (
    #         "The specified network must be either ``abelp'', "
    #         + "``apelp'', or ``auelp''."
    #     )
    #     print(error_str)
    #     return None
    # else:
    #     # Additional loading for GGML models once that code is complete!!!
    #     if aelp_network == "abelp":
    #         from src.models.abelp_networks_dataset import abelpDataset
    #         aelp_dataset = abelpDataset
    #     elif aelp_network == "apelp":
    #         from src.models.apelp_networks_dataset import apelpDataset
    #         aelp_dataset = apelpDataset
    #     elif aelp_network == "auelp":
    #         from src.models.auelp_networks_dataset import auelpDataset
    #         aelp_dataset = auelpDataset

    # train_dataset = aelp_dataset(cfg, "train")
    # val_dataset = aelp_dataset(cfg, "val")
    # test_dataset = aelp_dataset(cfg, "test")

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # # debugging sandbox
    # for data in tqdm(train_dataloader):
    #     print(data)
    #     break
    
    # train_data = data
    # print(train_data)
    # batch = train_data.batch
    # n = batch.size(0)
    # batch_size = torch.max(batch).item() + 1
    # n_graph = n // batch_size
    # print(n, batch_size, n_graph)

    # pos_edge_index_0 = train_data.edge_index_0
    # # num_neg_samples = int(cfg.model.neg_edge_graph_adj_rho*n_graph**2)
    # num_neg_samples = n_graph**2
    # print(num_neg_samples)

    # neg_edge_index = batched_negative_sampling(pos_edge_index_0, batch, num_neg_samples)
    # print(neg_edge_index.shape)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Training debugging code took {execution_time} seconds to run")