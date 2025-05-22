# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import filename_str
from src.networks.aelp_networks import aelp_filename_str
from plotting.plotting_utils import data_histogram_plotter
from src.networks.apelp_networks_config import sample_params_arr_func

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    _, sample_num = sample_params_arr_func(cfg)

    ##### Plot the topological descriptors for each artificial
    ##### polydisperse end-linked polymer network parameter sample
    print("Plot the topological descriptors", flush=True)

    for sample in range(sample_num):
        l_cntr_conn_edges = np.asarray([])
        lcl_k = np.asarray([], dtype=int)
        lcl_k_diff = np.asarray([], dtype=int)
        lcl_avrg_nn_k = np.asarray([])
        lcl_lcl_avrg_kappa = np.asarray([])
        lcl_l = np.asarray([])
        lcl_l_naive = np.asarray([])
        nrmlzd_lcl_l = np.asarray([])
        nrmlzd_lcl_l_naive = np.asarray([])
        eeel_glbl_mean_k = np.asarray([])
        glbl_prop_eeel_n = np.asarray([])
        glbl_prop_eeel_m = np.asarray([])
        eeel_glbl_mean_gamma = np.asarray([])
        glbl_n_fractal_dim = np.asarray([])
        glbl_xi_corr = np.asarray([])
        
        # Generate filenames
        filename = filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        
        l_cntr_conn_edges_plt_filename = (
            filename + "-l_cntr_conn_edges" + ".png"
        )
        lcl_k_plt_filename = filename + "-lcl-k" + ".png"
        lcl_k_diff_plt_filename = filename + "-lcl-k_diff" + ".png"
        lcl_avrg_nn_k_plt_filename = filename + "-lcl-avrg_nn_k" + ".png"
        lcl_lcl_avrg_kappa_plt_filename = (
            filename + "-lcl-lcl_avrg_kappa" + ".png"
        )
        lcl_l_plt_filename = filename + "-lcl-l" + ".png"
        lcl_l_naive_plt_filename = filename + "-lcl-l_naive" + ".png"
        nrmlzd_lcl_l_plt_filename = filename + "-nrmlzd-lcl-l" + ".png"
        nrmlzd_lcl_l_naive_plt_filename = (
            filename + "-nrmlzd-lcl-l_naive" + ".png"
        )
        eeel_glbl_mean_k_plt_filename = filename + "-eeel-glbl-mean-k" + ".png"
        glbl_prop_eeel_n_plt_filename = filename + "-glbl-prop_eeel_n" + ".png"
        glbl_prop_eeel_m_plt_filename = filename + "-glbl-prop_eeel_m" + ".png"
        eeel_glbl_mean_gamma_plt_filename = (
            filename + "-eeel-glbl-mean-gamma" + ".png"
        )
        glbl_n_fractal_dim_plt_filename = (
            filename + "-glbl-n_fractal_dim" + ".png"
        )
        glbl_xi_corr_plt_filename = filename + "-glbl-xi_corr" + ".png"
        
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            
            # Load in graph data
            graph = np.load(graph_filename)

            l_max = np.max(graph["L"])

            l_cntr_conn_edges = np.concatenate(
                (l_cntr_conn_edges, graph["l_cntr_conn_edges"]))
            lcl_k = np.concatenate((lcl_k, graph["lcl_k"]), dtype=int)
            lcl_k_diff = np.concatenate(
                (lcl_k_diff, graph["lcl_k_diff"]), dtype=int)
            lcl_avrg_nn_k = np.concatenate(
                (lcl_avrg_nn_k, graph["lcl_avrg_nn_k"]))
            lcl_lcl_avrg_kappa = np.concatenate(
                (lcl_lcl_avrg_kappa, graph["lcl_lcl_avrg_kappa"]))
            lcl_l = np.concatenate((lcl_l, graph["lcl_l"]))
            lcl_l_naive = np.concatenate((lcl_l_naive, graph["lcl_l_naive"]))
            nrmlzd_lcl_l = np.concatenate((nrmlzd_lcl_l, graph["lcl_l"]/l_max))
            nrmlzd_lcl_l_naive = np.concatenate(
                (nrmlzd_lcl_l_naive, graph["lcl_l_naive"]/l_max))
            eeel_glbl_mean_k = np.concatenate(
                (eeel_glbl_mean_k, graph["eeel_glbl_mean_k"]))
            glbl_prop_eeel_n = np.concatenate(
                (glbl_prop_eeel_n, graph["glbl_prop_eeel_n"]))
            glbl_prop_eeel_m = np.concatenate(
                (glbl_prop_eeel_m, graph["glbl_prop_eeel_m"]))
            eeel_glbl_mean_gamma = np.concatenate(
                (eeel_glbl_mean_gamma, graph["eeel_glbl_mean_gamma"]))
            glbl_n_fractal_dim = np.concatenate(
                (glbl_n_fractal_dim, graph["glbl_n_fractal_dim"]))
            glbl_xi_corr = np.concatenate((glbl_xi_corr, graph["glbl_xi_corr"]))
        
        # Plot histograms of graph data
        data_histogram_plotter(
            l_cntr_conn_edges, "auto", "l_cntr", l_cntr_conn_edges_plt_filename)
        data_histogram_plotter(
            lcl_k, np.arange(np.max(lcl_k)+2, dtype=int), "k", lcl_k_plt_filename)
        data_histogram_plotter(
            lcl_k_diff, np.arange(np.max(lcl_k_diff)+2, dtype=int), "k_diff",
            lcl_k_diff_plt_filename)
        data_histogram_plotter(
            lcl_avrg_nn_k, "auto", "avrg_nn_k", lcl_avrg_nn_k_plt_filename)
        data_histogram_plotter(
            lcl_lcl_avrg_kappa, "auto", "lcl_avrg_kappa",
            lcl_lcl_avrg_kappa_plt_filename)
        data_histogram_plotter(lcl_l, "auto", "l", lcl_l_plt_filename)
        data_histogram_plotter(
            lcl_l_naive, "auto", "l_naive", lcl_l_naive_plt_filename)
        data_histogram_plotter(
            nrmlzd_lcl_l, "auto", "l_nrmlzd", nrmlzd_lcl_l_plt_filename)
        data_histogram_plotter(
            nrmlzd_lcl_l_naive, "auto", "l_naive_nrmlzd",
            nrmlzd_lcl_l_naive_plt_filename)
        data_histogram_plotter(
            eeel_glbl_mean_k, "auto", "mean_k", eeel_glbl_mean_k_plt_filename)
        data_histogram_plotter(
            glbl_prop_eeel_n, "auto", "eeel_n/n", glbl_prop_eeel_n_plt_filename)
        data_histogram_plotter(
            glbl_prop_eeel_m, "auto", "eeel_m/m", glbl_prop_eeel_m_plt_filename)
        data_histogram_plotter(
            eeel_glbl_mean_gamma, "auto", "mean_gamma",
            eeel_glbl_mean_gamma_plt_filename)
        data_histogram_plotter(
            glbl_n_fractal_dim, "auto", "n_fractal_dim",
            glbl_n_fractal_dim_plt_filename)
        data_histogram_plotter(
            glbl_xi_corr, "auto", "xi_corr", glbl_xi_corr_plt_filename)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network topology descriptors plotting took {execution_time} seconds to run")