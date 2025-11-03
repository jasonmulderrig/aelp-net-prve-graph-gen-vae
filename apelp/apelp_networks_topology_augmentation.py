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
    run_aelp_network_hilbert_node_label_assignment,
    run_aelp_network_multiedge_order_segregation
)
from src.networks.aelp_networks import aelp_filename_str
from src.networks.apelp_networks_config import (
    params_list_func,
    sample_params_arr_func,
    sample_config_params_arr_func
)

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    # Gather arrays of configuration parameters
    _, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    
    ##### Perform the additional node seeding procedure for each
    ##### artificial polydisperse end-linked polymer network parameter
    ##### sample
    print("Performing the additional node seeding procedure", flush=True)

    # Initialize an array to store the number of nodes in the initial
    # network for each configuration
    n_coords = np.empty(sample_config_num, dtype=int)

    # Calculate maximum number of nodes in the initial network for each
    # sample
    indx = 0
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            coords_filename = (
                aelp_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
                + ".coords"
            )
            n_coords[indx] = np.shape(np.loadtxt(coords_filename, ndmin=1))[0]
            indx += 1
    n_coords_max = np.max(n_coords)
    
    # Populate the network sample configuration parameters array
    sample_config_addtnl_n_params_arr = sample_config_params_arr.copy()
    for indx in range(sample_config_num):
        sample_config_addtnl_n_params_arr[indx, 7] = n_coords_max
    sample_config_addtnl_n_params_arr = (
        sample_config_addtnl_n_params_arr[:, [0, 1, 2, 7, 10]]
    ) # sample, dim, b, n, config
    apelp_network_additional_node_seeding_params_list = params_list_func(
        sample_config_addtnl_n_params_arr)
    apelp_network_additional_node_seeding_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), float(b), int(n), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, n, config) in apelp_network_additional_node_seeding_params_list
        ]
    )
    random.shuffle(apelp_network_additional_node_seeding_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_node_seeding,
            apelp_network_additional_node_seeding_args)
    
    ##### Perform the additional nodes type procedure for each
    ##### artificial polydisperse end-linked polymer network parameter
    ##### sample
    print("Performing the additional nodes type procedure", flush=True)

    sample_config_addtnl_nodes_type_params_arr = (
        sample_config_addtnl_n_params_arr[:, [0, 3, 4]]
    ) # sample, n, config
    apelp_network_additional_nodes_type_params_list = params_list_func(
        sample_config_addtnl_nodes_type_params_arr)
    apelp_network_additional_nodes_type_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(n), int(config))
            for (sample, n, config) in apelp_network_additional_nodes_type_params_list
        ]
    )
    random.shuffle(apelp_network_additional_nodes_type_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_nodes_type,
            apelp_network_additional_nodes_type_args)
    
    ##### Reassign the node labels using the Hilbert space-filling curve
    ##### for each artificial polydisperse end-linked polymer network
    print(
        "Reassigning node labels using the Hilbert space-filling curve",
        flush=True)

    apelp_network_hilbert_node_label_assignment_params_arr = (
        sample_config_params_arr[:, [0, 10]]
    ) # sample, config
    apelp_network_hilbert_node_label_assignment_params_list = params_list_func(
        apelp_network_hilbert_node_label_assignment_params_arr)
    apelp_network_hilbert_node_label_assignment_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config))
            for (sample, config) in apelp_network_hilbert_node_label_assignment_params_list
        ]
    )
    random.shuffle(apelp_network_hilbert_node_label_assignment_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            apelp_network_hilbert_node_label_assignment_args)
    
    ##### Segregate edges of different multiedge order for each 
    ##### artificial polydisperse end-linked polymer network
    print("Segregating edges of different multiedge order", flush=True)

    apelp_network_multiedge_order_segregation_params_arr = (
        sample_config_params_arr[:, [0, 6, 10]]
    ) # sample, k, config
    apelp_network_multiedge_order_segregation_params_list = params_list_func(
        apelp_network_multiedge_order_segregation_params_arr)
    apelp_network_multiedge_order_segregation_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(k), int(config))
            for (sample, k, config) in apelp_network_multiedge_order_segregation_params_list
        ]
    )
    random.shuffle(apelp_network_multiedge_order_segregation_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_multiedge_order_segregation,
            apelp_network_multiedge_order_segregation_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network topology augmentation protocol took {execution_time} seconds to run")