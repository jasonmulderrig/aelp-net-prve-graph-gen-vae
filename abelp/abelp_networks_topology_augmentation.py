# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import multiprocessing
import random
import numpy as np
from src.helpers.multiprocessing_utils import (
    run_aelp_network_additional_node_seeding,
    run_aelp_network_additional_nodes_type,
    run_aelp_network_hilbert_node_label_assignment
)
from src.networks.aelp_networks import aelp_filename_str
from src.networks.abelp_networks_config import (
    params_list_func,
    sample_params_arr_func,
    sample_config_params_arr_func
)

@hydra.main(
        version_base=None,
        config_path="../configs/networks/abelp",
        config_name="abelp_networks")
def main(cfg: DictConfig) -> None:
    # Gather arrays of configuration parameters
    _, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    
    ##### Perform the additional node seeding procedure for each
    ##### artificial bimodal end-linked polymer network parameter sample
    print("Performing the additional node seeding procedure", flush=True)

    # Initialize an array to store the maximum number of nodes in the
    # initial network for each sample
    sample_n_coords_max = np.empty(sample_num, dtype=int)

    # Calculate maximum number of nodes in the initial network for each
    # sample
    for sample in range(sample_num):
        sample_n_coords = np.asarray([], dtype=int)
        for config in range(cfg.topology.config):
            coords_filename = (
                aelp_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename, ndmin=1)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])),
                dtype=int)
        sample_n_coords_max[sample] = np.max(sample_n_coords)
    
    # Populate the network sample configuration parameters array
    sample_config_addtnl_n_params_arr = sample_config_params_arr.copy()
    for indx in range(sample_config_num):
        sample = int(sample_config_addtnl_n_params_arr[indx, 0])
        sample_config_addtnl_n_params_arr[indx, 7] = sample_n_coords_max[sample]
    
    abelp_network_additional_node_seeding_params_list = params_list_func(
        sample_config_addtnl_n_params_arr)
    abelp_network_additional_node_seeding_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), float(b), int(n), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, _, _, _, _, n, _, _, _, config) in abelp_network_additional_node_seeding_params_list
        ]
    )
    random.shuffle(abelp_network_additional_node_seeding_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_node_seeding,
            abelp_network_additional_node_seeding_args)
    
    ##### Perform the additional nodes type procedure for each
    ##### artificial bimodal end-linked polymer network parameter sample
    print("Performing the additional nodes type procedure", flush=True)
    
    abelp_network_additional_nodes_type_params_list = (
        abelp_network_additional_node_seeding_params_list.copy()
    )
    abelp_network_additional_nodes_type_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(n), int(config))
            for (sample, _, _, _, _, _, _, n, _, _, _, config) in abelp_network_additional_nodes_type_params_list
        ]
    )
    random.shuffle(abelp_network_additional_nodes_type_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_nodes_type,
            abelp_network_additional_nodes_type_args)
    
    ##### Reassign the node labels using the Hilbert space-filling curve
    ##### for each artificial bimodal end-linked polymer network
    print(
        "Reassigning node labels using the Hilbert space-filling curve",
        flush=True)

    abelp_network_hilbert_node_label_assignment_params_arr = (
        sample_config_params_arr[:, [0, 11]]
    ) # sample, config
    abelp_network_hilbert_node_label_assignment_params_list = params_list_func(
        abelp_network_hilbert_node_label_assignment_params_arr)
    abelp_network_hilbert_node_label_assignment_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config))
            for (sample, config) in abelp_network_hilbert_node_label_assignment_params_list
        ]
    )
    random.shuffle(abelp_network_hilbert_node_label_assignment_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            abelp_network_hilbert_node_label_assignment_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial bimodal end-linked polymer network topology augmentation protocol took {execution_time} seconds to run")