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
from src.file_io.file_io import (
    config_filename_str,
    L_filename_str
)
from src.helpers.multiprocessing_utils import (
    run_aelp_L,
    run_initial_node_seeding,
    run_apelp_network_topology,
    run_aelp_network_additional_node_seeding,
    run_aelp_network_additional_nodes_type,
    run_aelp_network_hilbert_node_label_assignment,
    run_aelp_network_multiedge_order_segregation,
    run_aelp_network_local_topological_descriptor,
    run_aelp_network_local_multiedge_order_topological_descriptor,
    run_aelp_network_global_property_descriptor
)
from src.networks.aelp_networks import (
    aelp_filename_str,
    aelp_multiedge_max
)
from src.networks.apelp_networks_config import (
    params_list_func,
    sample_params_arr_func,
    sample_config_params_arr_func
)
from src.models.apelp_networks_dataset import valid_params

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    # Gather arrays of configuration parameters
    sample_params_arr, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)

    ##### Calculate and save L for each artificial polydisperse
    ##### end-linked polymer network parameter sample
    print("Calculating simulation box side lengths", flush=True)

    if sample_params_arr.ndim == 1:
        L_params_arr = (
            sample_params_arr[[0, 1, 4, 5, 6, 7, 8]]
        ) # sample, dim, chi, rho_en, k, n, en
    else:
        L_params_arr = (
            sample_params_arr[:, [0, 1, 4, 5, 6, 7, 8]]
        ) # sample, dim, chi, rho_en, k, n, en
    L_params_list = params_list_func(L_params_arr)
    L_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(dim), chi, rho_en, int(k), int(n), int(en))
            for (sample, dim, chi, rho_en, k, n, en) in L_params_list
        ]
    )
    random.shuffle(L_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_aelp_L, L_args)
    
    ##### Perform the initial node seeding procedure for each artificial
    ##### polydisperse end-linked polymer network parameter sample
    print("Performing the initial node seeding", flush=True)

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 7, 10]]
    ) # sample, dim, b, n, config
    initial_node_seeding_params_list = params_list_func(
        initial_node_seeding_params_arr)
    initial_node_seeding_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), b, int(n), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, n, config) in initial_node_seeding_params_list
        ]
    )
    random.shuffle(initial_node_seeding_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_initial_node_seeding, initial_node_seeding_args)
    
    # Check to see if the number of seeded nodes, prhd_n, equals the
    # intended/specified number of nodes to be seeded, n. Continue to
    # the topology initialization procedure ONLY IF prhd_n = n. If
    # prhd_n != n for any specified network, then the code block
    # identifies which particular set(s) of network parameters
    # prhd_n != n occurred for.
    if cfg.label.scheme == "prhd":
        prhd_n_vs_n = np.zeros(sample_config_num)
        
        for indx in range(sample_config_num):
            sample = int(sample_config_params_arr[indx, 0])
            n = int(sample_config_params_arr[indx, 7])
            config = int(sample_config_params_arr[indx, 10])
            
            coords_filename = (
                config_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename, ndmin=1)
            
            if np.shape(coords)[0] == n: prhd_n_vs_n[indx] = 1
            else: pass

        sample_config_params_prhd_n_neq_n = (
            sample_config_params_arr[np.where(prhd_n_vs_n == 0)]
        )
        
        if np.shape(sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n for all apelp network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of apelp network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of apelp "
                + "network parameters before continuing on to the "
                + "topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the network topology initialization procedure for each
    ##### polydisperse artificial end-linked polymer network parameter
    ##### sample
    print_str = (
        "Performing the artificial polydisperse end-linked polymer "
        + "network topology initialization procedure"
    )
    print(print_str, flush=True)

    topology_params_arr = (
        np.delete(sample_config_params_arr, 5, axis=1)
    ) # sample, dim, b, xi, chi, k, n, en, en_max, config
    topology_params_list = params_list_func(topology_params_arr)
    topology_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), b, xi, chi, int(k), int(n), int(en), int(en_max), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, xi, chi, k, n, en, en_max, config) in topology_params_list
        ]
    )
    random.shuffle(topology_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_apelp_network_topology, topology_args)
    
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

    ##### Calculate descriptors
    print("Calculating descriptors", flush=True)

    local_topological_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2]]
    ) # sample, config, b
    local_topological_descriptors_params_list = params_list_func(
        local_topological_descriptors_params_arr)
    local_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *lcl_tplgcl_dscrptrs)
            for (sample, config, b) in local_topological_descriptors_params_list
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_topological_descriptor_args)

    local_multiedge_order_topological_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2, 6]]
    ) # sample, config, b, k
    local_multiedge_order_topological_descriptors_params_list = params_list_func(
        local_multiedge_order_topological_descriptors_params_arr)
    local_multiedge_order_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, int(k), *lcl_tplgcl_dscrptrs)
            for (sample, config, b, k) in local_multiedge_order_topological_descriptors_params_list
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_multiedge_order_topological_descriptor_args)

    global_property_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2]]
    ) # sample, config, b
    global_property_descriptors_params_list = params_list_func(
        global_property_descriptors_params_arr)
    global_property_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_prprty_dscrptrs)
            for (sample, config, b) in global_property_descriptors_params_list
            for glbl_prprty_dscrptrs in list(map(tuple, cfg.descriptors.global_property_descriptors))
        ]
    )
    random.shuffle(global_property_descriptor_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_local_topological_descriptor,
            local_topological_descriptor_args)
        pool.map(
            run_aelp_network_local_multiedge_order_topological_descriptor,
            local_multiedge_order_topological_descriptor_args)
        pool.map(
            run_aelp_network_global_property_descriptor,
            global_property_descriptor_args)
    
    ##### Gather and save data components for each sample in .npz files
    print(
        "Gather and save data components for each sample in .npz files",
        flush=True)
    
    # Validate artificial polydisperse end-linked polymer network
    # topology parameters
    dim, b, rho_en, k, n, en_max, multiedge_max = valid_params(cfg)
    
    # Calculate the mean simulation box side lengths
    L = np.empty((sample_num, dim))
    for sample in range(sample_num):
        # Load in simulation box side lengths
        L_filename = L_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        L[sample] = np.loadtxt(L_filename, ndmin=1)
    L_mean = np.mean(L, axis=0)

    # Gather and save data components
    for indx in range(sample_config_num):
        # Initialize graph data and filenames dictionaries
        graph_data = {}
        graph_data_filenames = {}

        # Gather sample and config number
        sample = int(sample_config_params_arr[indx, 0])
        config = int(sample_config_params_arr[indx, 10])

        # Collect the synthesis processing parameter values
        graph_data["xi"] = np.asarray([sample_config_params_arr[indx, 3]])
        graph_data["chi"] = np.asarray([sample_config_params_arr[indx, 4]])
        graph_data["en_mean"] = np.asarray([sample_config_params_arr[indx, 8]])

        # Load in simulation box side lengths, and collect simulation
        # box side lengths and the mean simulation box side lengths
        L_filename = L_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        graph_data["L"] = np.loadtxt(L_filename, ndmin=1)
        graph_data["L_mean"] = L_mean

        # Generate filenames
        aelp_filename = aelp_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample,
            config)
        graph_filename = aelp_filename + ".npz"
        
        graph_data_filenames["orgnl_coords"] = (
            aelp_filename + "_orgnl" + ".coords"
        )
        graph_data_filenames["coords"] = aelp_filename + ".coords"
        graph_data_filenames["core_nodes_type"] = (
            aelp_filename + "-core_nodes_type" + ".dat"
        )
        graph_data_filenames["conn_edges"] = (
            aelp_filename + "-conn_edges" + ".dat"
        )
        graph_data_filenames["conn_edges_type"] = (
            aelp_filename + "-conn_edges_type" + ".dat"
        )
        graph_data_filenames["l_cntr_conn_edges"] = (
            aelp_filename + "-l_cntr_conn_edges" + ".dat"
        )
        graph_data_filenames["lcl_k"] = aelp_filename + "-lcl-k" + ".dat"
        graph_data_filenames["lcl_k_diff"] = (
            aelp_filename + "-lcl-k_diff" + ".dat"
        )
        graph_data_filenames["lcl_c"] = aelp_filename + "-lcl-c" + ".dat"
        graph_data_filenames["lcl_avrg_nn_k"] = (
            aelp_filename + "-lcl-avrg_nn_k" + ".dat"
        )
        graph_data_filenames["lcl_lcl_avrg_kappa"] = (
            aelp_filename + "-lcl-lcl_avrg_kappa" + ".dat"
        )
        graph_data_filenames["lcl_l"] = aelp_filename + "-lcl-l" + ".dat"
        graph_data_filenames["lcl_l_naive"] = (
            aelp_filename + "-lcl-l_naive" + ".dat"
        )
        for multiedge in range(multiedge_max):
            graph_data_filenames["conn_edges_"+str(multiedge)] = (
                aelp_filename + "-conn_edges_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["conn_edges_type_"+str(multiedge)] = (
                aelp_filename + "-conn_edges_type_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["l_cntr_conn_edges_"+str(multiedge)] = (
                aelp_filename + "-l_cntr_conn_edges_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_k_"+str(multiedge)] = (
                aelp_filename + "-lcl-k_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_k_diff_"+str(multiedge)] = (
                aelp_filename + "-lcl-k_diff_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_c_"+str(multiedge)] = (
                aelp_filename + "-lcl-c_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_avrg_nn_k_"+str(multiedge)] = (
                aelp_filename + "-lcl-avrg_nn_k_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_lcl_avrg_kappa_"+str(multiedge)] = (
                aelp_filename + "-lcl-lcl_avrg_kappa_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_l_"+str(multiedge)] = (
                aelp_filename + "-lcl-l_" + str(multiedge) + ".dat"
            )
            graph_data_filenames["lcl_l_naive_"+str(multiedge)] = (
                aelp_filename + "-lcl-l_naive_" + str(multiedge) + ".dat"
            )
        graph_data_filenames["eeel_dobrynin_kappa"] = (
            aelp_filename + "-eeel_dobrynin_kappa" + ".dat"
        )
        graph_data_filenames["eeel_glbl_mean_gamma"] = (
            aelp_filename + "-eeel_glbl_mean_gamma" + ".dat"
        )
        
        # Load in graph data
        graph_data["orgnl_coords"] = np.loadtxt(
            graph_data_filenames["orgnl_coords"], ndmin=1)
        graph_data["coords"] = np.loadtxt(
            graph_data_filenames["coords"], ndmin=1)
        graph_data["core_nodes_type"] = np.loadtxt(
            graph_data_filenames["core_nodes_type"], dtype=int, ndmin=1)
        graph_data["conn_edges"] = np.loadtxt(
            graph_data_filenames["conn_edges"], dtype=int, ndmin=1)
        graph_data["conn_edges_type"] = np.loadtxt(
            graph_data_filenames["conn_edges_type"], dtype=int, ndmin=1)
        graph_data["l_cntr_conn_edges"] = np.loadtxt(
            graph_data_filenames["l_cntr_conn_edges"], ndmin=1)
        graph_data["lcl_k"] = np.loadtxt(
            graph_data_filenames["lcl_k"], dtype=int, ndmin=1)
        graph_data["lcl_k_diff"] = np.loadtxt(
            graph_data_filenames["lcl_k_diff"], dtype=int, ndmin=1)
        graph_data["lcl_c"] = np.loadtxt(graph_data_filenames["lcl_c"], ndmin=1)
        graph_data["lcl_avrg_nn_k"] = np.loadtxt(
            graph_data_filenames["lcl_avrg_nn_k"], ndmin=1)
        graph_data["lcl_lcl_avrg_kappa"] = np.loadtxt(
            graph_data_filenames["lcl_lcl_avrg_kappa"], ndmin=1)
        graph_data["lcl_l"] = np.loadtxt(graph_data_filenames["lcl_l"], ndmin=1)
        graph_data["lcl_l_naive"] = np.loadtxt(
            graph_data_filenames["lcl_l_naive"], ndmin=1)
        for multiedge in range(multiedge_max):
            graph_data["conn_edges_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["conn_edges_"+str(multiedge)], dtype=int,
                ndmin=1)
            graph_data["conn_edges_type_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["conn_edges_type_"+str(multiedge)],
                dtype=int, ndmin=1)
            graph_data["l_cntr_conn_edges_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["l_cntr_conn_edges_"+str(multiedge)],
                ndmin=1)
            graph_data["lcl_k_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_k_"+str(multiedge)], dtype=int,
                ndmin=1)
            graph_data["lcl_k_diff_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_k_diff_"+str(multiedge)], dtype=int,
                ndmin=1)
            graph_data["lcl_c_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_c_"+str(multiedge)], ndmin=1)
            graph_data["lcl_avrg_nn_k_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_avrg_nn_k_"+str(multiedge)], ndmin=1)
            graph_data["lcl_lcl_avrg_kappa_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_lcl_avrg_kappa_"+str(multiedge)],
                ndmin=1)
            graph_data["lcl_l_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_l_"+str(multiedge)], ndmin=1)
            graph_data["lcl_l_naive_"+str(multiedge)] = np.loadtxt(
                graph_data_filenames["lcl_l_naive_"+str(multiedge)], ndmin=1)
        graph_data["eeel_dobrynin_kappa"] = np.loadtxt(
            graph_data_filenames["eeel_dobrynin_kappa"], ndmin=1)
        graph_data["eeel_glbl_mean_gamma"] = np.loadtxt(
            graph_data_filenames["eeel_glbl_mean_gamma"], ndmin=1)

        # Save graph data in an .npz file
        np.savez(graph_filename, **graph_data)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"One shot artificial polydisperse end-linked polymer network data creation protocol took {execution_time} seconds to run")