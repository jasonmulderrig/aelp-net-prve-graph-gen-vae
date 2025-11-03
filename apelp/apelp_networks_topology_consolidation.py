# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import L_filename_str
from src.networks.aelp_networks import (
    aelp_filename_str,
    aelp_multiedge_max
)
from src.networks.apelp_networks_config import (
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
    _, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)

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
    print(f"Artificial polydisperse end-linked polymer network data consolidation protocol took {execution_time} seconds to run")