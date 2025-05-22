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
    L_filename_str,
    config_filename_str
)
from src.helpers.multiprocessing_utils import (
    run_aelp_L,
    run_initial_node_seeding,
    run_auelp_network_topology,
    run_aelp_network_additional_node_seeding,
    run_aelp_network_additional_nodes_type,
    run_aelp_network_hilbert_node_label_assignment,
    run_aelp_network_local_topological_descriptor,
    run_aelp_network_global_topological_descriptor,
    run_aelp_network_global_morphological_descriptor
)
from src.networks.auelp_networks_config import (
    params_list_func,
    params_arr_func,
    sample_params_arr_func,
    sample_config_params_arr_func
)
from src.networks.aelp_networks import aelp_filename_str

@hydra.main(
        version_base=None,
        config_path="../configs/networks/auelp",
        config_name="auelp_networks")
def main(cfg: DictConfig) -> None:
    # Gather arrays of configuration parameters
    _, sample_num = params_arr_func(cfg)
    sample_params_arr, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    b = cfg.topology.b[0]

    ##### Calculate and save L for each artificial uniform end-linked
    ##### polymer network parameter sample
    print("Calculating simulation box side lengths", flush=True)

    if sample_params_arr.ndim == 1:
        L_params_arr = (
            sample_params_arr[[0, 1, 4, 5, 6, 7]]
        ) # sample, dim, rho_en, k, n, en
    else:
        L_params_arr = (
            sample_params_arr[:, [0, 1, 4, 5, 6, 7]]
        ) # sample, dim, rho_en, k, n, en
    L_params_list = params_list_func(L_params_arr)
    L_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(dim), rho_en, int(k), int(n), int(en))
            for (sample, dim, rho_en, k, n, en) in L_params_list
        ]
    )
    random.shuffle(L_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_aelp_L, L_args)
        
    ##### Perform the initial node seeding procedure for each artificial
    ##### uniform end-linked polymer network parameter sample
    print("Performing the initial node seeding", flush=True)

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 6, 8]]
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
            n = int(sample_config_params_arr[indx, 6])
            config = int(sample_config_params_arr[indx, 8])
            
            coords_filename = (
                config_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n: prhd_n_vs_n[indx] = 1
            else: pass

        sample_config_params_prhd_n_neq_n = (
            sample_config_params_arr[np.where(prhd_n_vs_n == 0)]
        )
        
        if np.shape(sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = (
                "Success! prhd_n = n  for all auelp network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of auelp'' network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of auelp "
                + "network parameters before continuing on to the "
                + "topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the network topology initialization procedure for each
    ##### artificial uniform end-linked polymer network parameter sample
    print_str = (
        "Performing the artificial uniform end-linked polymer network "
        + "topology initialization procedure"
    )
    print(print_str, flush=True)

    topology_params_arr = (
        np.delete(sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, en, config
    topology_params_list = params_list_func(topology_params_arr)
    topology_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), b, xi, int(k), int(n), int(en), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, xi, k, n, en, config) in topology_params_list
        ]
    )
    random.shuffle(topology_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_auelp_network_topology, topology_args)
    
    ##### Perform the additional node seeding procedure for each
    ##### artificial uniform end-linked polymer network parameter sample
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
            coords = np.loadtxt(coords_filename)
            sample_n_coords = np.concatenate(
                (sample_n_coords, np.asarray([np.shape(coords)[0]])),
                dtype=int)
        sample_n_coords_max[sample] = np.max(sample_n_coords)
    
    # Populate the network sample configuration parameters array
    sample_config_addtnl_n_params_arr = sample_config_params_arr.copy()
    for indx in range(sample_config_num):
        sample = int(sample_config_addtnl_n_params_arr[indx, 0])
        sample_config_addtnl_n_params_arr[indx, 6] = sample_n_coords_max[sample]
    
    auelp_network_additional_node_seeding_params_list = params_list_func(
        sample_config_addtnl_n_params_arr)
    auelp_network_additional_node_seeding_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), float(b), int(n), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, _, _, _, n, _, config) in auelp_network_additional_node_seeding_params_list
        ]
    )
    random.shuffle(auelp_network_additional_node_seeding_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_node_seeding,
            auelp_network_additional_node_seeding_args)
    
    ##### Perform the additional nodes type procedure for each
    ##### artificial uniform end-linked polymer network parameter sample
    print("Performing the additional nodes type procedure", flush=True)
    
    auelp_network_additional_nodes_type_params_list = (
        auelp_network_additional_node_seeding_params_list.copy()
    )
    auelp_network_additional_nodes_type_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(n), int(config))
            for (sample, _, _, _, _, _, n, _, config) in auelp_network_additional_nodes_type_params_list
        ]
    )
    random.shuffle(auelp_network_additional_nodes_type_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_additional_nodes_type,
            auelp_network_additional_nodes_type_args)
    
    ##### Reassign the node labels using the Hilbert space-filling curve
    ##### for each artificial uniform end-linked polymer network
    print(
        "Reassigning node labels using the Hilbert space-filling curve",
        flush=True)

    auelp_network_hilbert_node_label_assignment_params_arr = (
        sample_config_params_arr[:, [0, 8]]
    ) # sample, config
    auelp_network_hilbert_node_label_assignment_params_list = params_list_func(
        auelp_network_hilbert_node_label_assignment_params_arr)
    auelp_network_hilbert_node_label_assignment_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config))
            for (sample, config) in auelp_network_hilbert_node_label_assignment_params_list
        ]
    )
    random.shuffle(auelp_network_hilbert_node_label_assignment_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_hilbert_node_label_assignment,
            auelp_network_hilbert_node_label_assignment_args)
    
    ##### Calculate descriptors
    print("Calculating descriptors", flush=True)

    local_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *lcl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_topological_descriptor_args)

    global_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for glbl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_topological_descriptors))
        ]
    )
    random.shuffle(global_topological_descriptor_args)

    global_morphological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_mrphlgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for glbl_mrphlgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_morphological_descriptors))
        ]
    )
    random.shuffle(global_morphological_descriptor_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_local_topological_descriptor,
            local_topological_descriptor_args)
        pool.map(
            run_aelp_network_global_topological_descriptor,
            global_topological_descriptor_args)
        pool.map(
            run_aelp_network_global_morphological_descriptor,
            global_morphological_descriptor_args)
    
    ##### Gather and save data components for each sample in .npz files
    print("Gather and save data components for each sample in .npz files", flush=True)

    for sample in range(sample_num):
        # Generate filenames
        L_filename = L_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            orgnl_coords_filename = aelp_filename + "_orgnl" + ".coords"
            coords_filename = aelp_filename + ".coords"
            core_nodes_type_filename = (
                aelp_filename + "-core_nodes_type" + ".dat"
            )
            conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
            conn_edges_type_filename = (
                aelp_filename + "-conn_edges_type" + ".dat"
            )
            l_cntr_conn_edges_filename = (
                aelp_filename + "-l_cntr_conn_edges" + ".dat"
            )
            lcl_k_filename = aelp_filename + "-lcl-k" + ".dat"
            lcl_k_diff_filename = aelp_filename + "-lcl-k_diff" + ".dat"
            lcl_avrg_nn_k_filename = aelp_filename + "-lcl-avrg_nn_k" + ".dat"
            lcl_lcl_avrg_kappa_filename = (
                aelp_filename + "-lcl-lcl_avrg_kappa" + ".dat"
            )
            lcl_l_filename = aelp_filename + "-lcl-l" + ".dat"
            lcl_l_naive_filename = aelp_filename + "-lcl-l_naive" + ".dat"
            eeel_glbl_mean_k_filename = (
                aelp_filename + "-eeel-glbl-mean-k" + ".dat"
            )
            glbl_prop_eeel_n_filename = (
                aelp_filename + "-glbl-prop_eeel_n" + ".dat"
            )
            glbl_prop_eeel_m_filename = (
                aelp_filename + "-glbl-prop_eeel_m" + ".dat"
            )
            eeel_glbl_mean_gamma_filename = (
                aelp_filename + "-eeel-glbl-mean-gamma" + ".dat"
            )
            glbl_n_fractal_dim_filename = (
                aelp_filename + "-glbl-n_fractal_dim" + ".dat"
            )
            glbl_xi_corr_filename = aelp_filename + "-glbl-xi_corr" + ".dat"
            graph_filename = aelp_filename + ".npz"

            # Load in graph data
            L = np.loadtxt(L_filename)
            orgnl_coords = np.loadtxt(orgnl_coords_filename)
            coords = np.loadtxt(coords_filename)
            core_nodes_type = np.loadtxt(core_nodes_type_filename, dtype=int)
            conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
            conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
            l_cntr_conn_edges = np.loadtxt(l_cntr_conn_edges_filename)
            lcl_k = np.loadtxt(lcl_k_filename, dtype=int)
            lcl_k_diff = np.loadtxt(lcl_k_diff_filename, dtype=int)
            lcl_avrg_nn_k = np.loadtxt(lcl_avrg_nn_k_filename)
            lcl_lcl_avrg_kappa = np.loadtxt(lcl_lcl_avrg_kappa_filename)
            lcl_l = np.loadtxt(lcl_l_filename)
            lcl_l_naive = np.loadtxt(lcl_l_naive_filename)
            eeel_glbl_mean_k = np.asarray(
                [np.loadtxt(eeel_glbl_mean_k_filename)])
            glbl_prop_eeel_n = np.asarray(
                [np.loadtxt(glbl_prop_eeel_n_filename)])
            glbl_prop_eeel_m = np.asarray(
                [np.loadtxt(glbl_prop_eeel_m_filename)])
            eeel_glbl_mean_gamma = np.asarray(
                [np.loadtxt(eeel_glbl_mean_gamma_filename)])
            glbl_n_fractal_dim = np.asarray(
                [np.loadtxt(glbl_n_fractal_dim_filename)])
            glbl_xi_corr = np.asarray([np.loadtxt(glbl_xi_corr_filename)])

            # Save graph data in an .npz file
            np.savez(
                graph_filename, L=L, orgnl_coords=orgnl_coords, coords=coords,
                core_nodes_type=core_nodes_type, conn_edges=conn_edges,
                conn_edges_type=conn_edges_type,
                l_cntr_conn_edges=l_cntr_conn_edges, lcl_k=lcl_k,
                lcl_k_diff=lcl_k_diff, lcl_avrg_nn_k=lcl_avrg_nn_k,
                lcl_lcl_avrg_kappa=lcl_lcl_avrg_kappa, lcl_l=lcl_l,
                lcl_l_naive=lcl_l_naive, eeel_glbl_mean_k=eeel_glbl_mean_k,
                glbl_prop_eeel_n=glbl_prop_eeel_n,
                glbl_prop_eeel_m=glbl_prop_eeel_m,
                eeel_glbl_mean_gamma=eeel_glbl_mean_gamma,
                glbl_n_fractal_dim=glbl_n_fractal_dim,
                glbl_xi_corr=glbl_xi_corr)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"One shot artificial uniform end-linked polymer network data creation protocol took {execution_time} seconds to run")