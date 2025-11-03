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
    if en_mean == 6.0:
        edgecolor = "black"
        label = "$\\bar\\nu = 5$"
    if en_mean == 11.0:
        edgecolor = "gray"
        label = "$\\bar\\nu = 10$"
    elif en_mean == 51.0:
        edgecolor = "brown"
        label = "$\\bar\\nu = 50$"
    return edgecolor, label

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    sample_params_arr, sample_num = sample_params_arr_func(cfg)

    glbl_rho_graph_vals = np.empty(sample_num*cfg.topology.config)
    indx = 0
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            npz_graph = np.load(graph_filename)
            glbl_rho_graph_vals[indx] = npz_graph["glbl_rho_graph"][0]
            indx += 1
    print("min(rho_graph) = {}, max(rho_graph) = {}, mean(rho_graph) = {}".format(np.min(glbl_rho_graph_vals), np.max(glbl_rho_graph_vals), np.mean(glbl_rho_graph_vals)))
    
    filepath = filepath_str(cfg.label.network)
    synthesis_rho_graph_space_fig_filename = (
        filepath + "synthesis_rho_graph_space" + ".png"
    )
    synthesis_property_space_fig_filename = (
        filepath + "synthesis_property_space" + ".png"
    )
    synthesis_property_space_legend_filename = (
        filepath + "synthesis_property_space_legend" + ".png"
    )

    rng = np.random.default_rng()

    fig = plt.figure()
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            npz_graph = np.load(graph_filename)

            glbl_rho_graph = npz_graph["glbl_rho_graph"]
            jitter = rng.random()
            xi = npz_graph["xi"]
            chi = npz_graph["chi"]
            en_mean = npz_graph["en_mean"]

            shape, _ = processing_protocol_marker_shape(xi)
            color, _ = processing_protocol_marker_color(chi)
            edgecolor, _ = processing_protocol_marker_edgecolor(1.0*en_mean)

            plt.scatter(
                glbl_rho_graph, jitter, c=color, marker=shape,
                edgecolors=edgecolor, s=100, linewidth=2)
    plt.grid(True, alpha=0.25)
    # plt.xlim()
    # plt.ylim()
    plt.xticks(
        [0.020, 0.022, 0.024, 0.026, 0.028, 0.030, 0.032, 0.034],
        ["$0.020$", "$0.022$", "$0.024$", "$0.026$", "$0.028$", "$0.030$", "$0.032$", "$0.034$"],
        fontsize=14)
    plt.yticks([])
    plt.xlabel("$\\rho_{\\text{graph}}$", fontsize=20)
    fig.tight_layout()
    fig.savefig(synthesis_rho_graph_space_fig_filename)
    plt.close()
    
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
                c=color, marker=shape, edgecolors=edgecolor, s=100, linewidth=2)
    plt.grid(True, alpha=0.25)
    plt.xticks(
        [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        ["$0.15$", "$0.20$", "$0.25$", "$0.30$", "$0.35$", "$0.40$", "$0.45$", "$0.50$"],
        fontsize=14)
    plt.yticks(
        [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36],
        ["$0.20$", "$0.22$", "$0.24$", "$0.26$", "$0.28$", "$0.30$", "$0.32$", "$0.34$", "$0.36$"],
        fontsize=14)
    plt.xlabel("$\\Gamma$", fontsize=20)
    plt.ylabel("$\\kappa$", fontsize=20)
    fig.tight_layout()
    fig.savefig(synthesis_property_space_fig_filename)
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
    print(f"Artificial polydisperse end-linked polymer network synthesis-property space plotting took {execution_time} seconds to run")