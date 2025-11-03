# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "cm"
from src.file_io.file_io import filepath_str
from src.networks.aelp_networks import aelp_filename_str
from src.networks.apelp_networks_config import sample_params_arr_func

def processing_protocol_marker_shape(xi: float) -> tuple[str, str]:
    if xi == 0.9:
        shape = "o"
        label = "$\\xi = 0.9$"
    elif xi == 0.925:
        shape = "s"
        label = "$\\xi = 0.925$"
    elif xi == 0.95:
        shape = "^"
        label = "$\\xi = 0.95$"
    return shape, label

def processing_protocol_marker_color(chi: float) -> tuple[str, str]:
    if chi == 0.9:
        color = "magenta"
        label = "$\\chi = 0.9$"
    elif chi == 1.0:
        color = "blue"
        label = "$\\chi = 1.0$"
    elif chi == 1.1:
        color = "green"
        label = "$\\chi = 1.1$"
    return color, label

def processing_protocol_marker_edgecolor(en_mean: float) -> tuple[str, str]:
    if en_mean == 4.0:
        edgecolor = "black"
        label = "$\\bar\\nu = 5$"
    if en_mean == 9.0:
        edgecolor = "gray"
        label = "$\\bar\\nu = 10$"
    elif en_mean == 49.0:
        edgecolor = "brown"
        label = "$\\bar\\nu = 50$"
    return edgecolor, label

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    sample_params_arr, sample_num = sample_params_arr_func(cfg)

    filepath = filepath_str("analysis")
    synthesis_property_space_fig_filename = (
        filepath + "synthesis_property_space" + ".png"
    )
    synthesis_property_space_averaged_fig_filename = (
        filepath + "synthesis_property_space_averaged" + ".png"
    )
    synthesis_property_space_legend_filename = (
        filepath + "synthesis_property_space_legend" + ".png"
    )
    
    fig = plt.figure()
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            npz_graph = np.load(graph_filename)

            eeel_dobrynin_kappa = npz_graph["eeel_dobrynin_kappa"]
            eeel_glbl_mean_gamma = npz_graph["eeel_glbl_mean_gamma"]
            xi = npz_graph["xi"]
            chi = npz_graph["chi"]
            en_mean = npz_graph["en_mean"]

            shape, _ = processing_protocol_marker_shape(xi)
            color, _ = processing_protocol_marker_color(chi)
            edgecolor, _ = processing_protocol_marker_edgecolor(1.0*en_mean)

            plt.scatter(
                eeel_glbl_mean_gamma, eeel_dobrynin_kappa,
                c=color, marker=shape, edgecolors=edgecolor, s=40, linewidth=1.1) # s=100, linewidth=2
    plt.grid(True, alpha=0.25)
    plt.xlim([0.10, 0.55])
    plt.xticks(
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        ["$0.10$", "$0.15$", "$0.20$", "$0.25$", "$0.30$", "$0.35$", "$0.40$", "$0.45$", "$0.50$", "$0.55$"],
        fontsize=14)
    plt.ylim([0.20, 0.38])
    plt.yticks(
        [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38],
        ["$0.20$", "$0.22$", "$0.24$", "$0.26$", "$0.28$", "$0.30$", "$0.32$", "$0.34$", "$0.36$", "$0.38$"],
        fontsize=14)
    plt.xlabel("$\\Gamma$", fontsize=20)
    plt.ylabel("$\\kappa$", fontsize=20)
    fig.tight_layout()
    fig.savefig(synthesis_property_space_fig_filename)
    plt.close()

    num_config_avrg = 10
    assert cfg.topology.config % num_config_avrg == 0
    eeel_dobrynin_kappa_sum = 0.0
    eeel_glbl_mean_gamma_sum = 0.0

    fig = plt.figure()
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            npz_graph = np.load(graph_filename)

            eeel_dobrynin_kappa_sum += npz_graph["eeel_dobrynin_kappa"]
            eeel_glbl_mean_gamma_sum += npz_graph["eeel_glbl_mean_gamma"]
            
            if (config+1)%num_config_avrg == 0:
                eeel_dobrynin_kappa_avrg = (
                    eeel_dobrynin_kappa_sum / num_config_avrg
                )
                eeel_glbl_mean_gamma_avrg = (
                    eeel_glbl_mean_gamma_sum / num_config_avrg
                )
                xi = npz_graph["xi"]
                chi = npz_graph["chi"]
                en_mean = npz_graph["en_mean"]

                shape, _ = processing_protocol_marker_shape(xi)
                color, _ = processing_protocol_marker_color(chi)
                edgecolor, _ = processing_protocol_marker_edgecolor(1.0*en_mean)

                plt.scatter(
                    eeel_glbl_mean_gamma_avrg, eeel_dobrynin_kappa_avrg,
                    c=color, marker=shape, edgecolors=edgecolor, s=40,
                    linewidth=1.1) # s=100, linewidth=2

                eeel_dobrynin_kappa_sum = 0.0
                eeel_glbl_mean_gamma_sum = 0.0
    plt.grid(True, alpha=0.25)
    plt.xlim([0.10, 0.55])
    plt.xticks(
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        ["$0.10$", "$0.15$", "$0.20$", "$0.25$", "$0.30$", "$0.35$", "$0.40$", "$0.45$", "$0.50$", "$0.55$"],
        fontsize=14)
    plt.ylim([0.20, 0.38])
    plt.yticks(
        [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38],
        ["$0.20$", "$0.22$", "$0.24$", "$0.26$", "$0.28$", "$0.30$", "$0.32$", "$0.34$", "$0.36$", "$0.38$"],
        fontsize=14)
    plt.xlabel("$\\Gamma$", fontsize=20)
    plt.ylabel("$\\kappa$", fontsize=20)
    fig.tight_layout()
    fig.savefig(synthesis_property_space_averaged_fig_filename)
    plt.close()

    fig = plt.figure()
    fig_legend = plt.figure()
    ax = fig.add_subplot(111)
    for xi_val in np.nditer(np.unique(sample_params_arr[:, 3])):
        shape, label = processing_protocol_marker_shape(xi_val)
        ax.scatter([], [], c="black", marker=shape, s=100, label=label)
    for chi_val in np.nditer(np.unique(sample_params_arr[:, 4])):
        color, label = processing_protocol_marker_color(chi_val)
        ax.scatter([], [], c=color, s=100, label=label)
    for en_mean_val in np.nditer(np.unique(sample_params_arr[:, 8])):
        edgecolor, label = processing_protocol_marker_edgecolor(en_mean_val)
        ax.scatter(
            [], [], c="white", edgecolors=edgecolor, s=100, linewidth=2,
            label=label)
    fig_legend.legend(
        ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1],
        loc="center", fontsize=20)
    fig_legend.tight_layout()
    fig_legend.savefig(synthesis_property_space_legend_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Synthesis-property space plotting took {execution_time} seconds to run")