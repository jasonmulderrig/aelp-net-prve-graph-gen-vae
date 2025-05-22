# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import L_filename_str
from src.networks.aelp_networks import aelp_filename_str
from src.networks.abelp_networks_config import sample_params_arr_func

@hydra.main(
        version_base=None,
        config_path="../configs/networks/abelp",
        config_name="abelp_networks")
def main(cfg: DictConfig) -> None:
    _, sample_num = sample_params_arr_func(cfg)
    
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
    print(f"Artificial bimodal end-linked polymer network data consolidation protocol took {execution_time} seconds to run")