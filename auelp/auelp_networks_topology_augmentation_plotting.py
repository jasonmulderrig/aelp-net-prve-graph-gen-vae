# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from src.file_io.file_io import filename_str
from src.networks.aelp_networks import aelp_filename_str

@hydra.main(
        version_base=None,
        config_path="../configs/networks/auelp",
        config_name="auelp_networks")
def main(cfg: DictConfig) -> None:
    ##### Plot the node label localization
    print("Plotting the node label localization", flush=True)

    for dim in cfg.topology.dim:
        sample = dim - 2 # dim - 3
        orgnl_coords = []
        coords = []
        nrmlzd_orgnl_coords = []
        nrmlzd_coords = []

        # Generate filenames
        filename = filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        for config in range(cfg.topology.config):
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample,
                config)
            graph_filename = aelp_filename + ".npz"
            
            # Load in graph data
            graph = np.load(graph_filename)

            # Gather and normalize coordinates
            L = graph["L"]
            orgnl_coords.append(graph["orgnl_coords"])
            coords.append(graph["coords"])
            nrmlzd_orgnl_coords.append(graph["orgnl_coords"]/L)
            nrmlzd_coords.append(graph["coords"]/L)
        
        orgnl_coords = np.asarray(orgnl_coords)
        coords = np.asarray(coords)
        nrmlzd_orgnl_coords = np.asarray(nrmlzd_orgnl_coords)
        nrmlzd_coords = np.asarray(nrmlzd_coords)
        n = np.shape(orgnl_coords)[1]

        # Calculate statistics
        orgnl_coords_min = np.empty((n, dim))
        orgnl_coords_mean_minus_std = np.empty((n, dim))
        orgnl_coords_mean = np.empty((n, dim))
        orgnl_coords_mean_plus_std = np.empty((n, dim))
        orgnl_coords_max = np.empty((n, dim))

        coords_min = np.empty((n, dim))
        coords_mean_minus_std = np.empty((n, dim))
        coords_mean = np.empty((n, dim))
        coords_mean_plus_std = np.empty((n, dim))
        coords_max = np.empty((n, dim))

        nrmlzd_orgnl_coords_min = np.empty((n, dim))
        nrmlzd_orgnl_coords_mean_minus_std = np.empty((n, dim))
        nrmlzd_orgnl_coords_mean = np.empty((n, dim))
        nrmlzd_orgnl_coords_mean_plus_std = np.empty((n, dim))
        nrmlzd_orgnl_coords_max = np.empty((n, dim))

        nrmlzd_coords_min = np.empty((n, dim))
        nrmlzd_coords_mean_minus_std = np.empty((n, dim))
        nrmlzd_coords_mean = np.empty((n, dim))
        nrmlzd_coords_mean_plus_std = np.empty((n, dim))
        nrmlzd_coords_max = np.empty((n, dim))

        for node_label in range(n):
            for coord in range(dim):
                mean_val = np.mean(orgnl_coords[:, node_label, coord])
                std_val = np.std(orgnl_coords[:, node_label, coord])
                orgnl_coords_min[node_label, coord] = np.min(
                    orgnl_coords[:, node_label, coord])
                orgnl_coords_mean_minus_std[node_label, coord] = (
                    mean_val - std_val
                )
                orgnl_coords_mean[node_label, coord] = mean_val
                orgnl_coords_mean_plus_std[node_label, coord] = (
                    mean_val + std_val
                )
                orgnl_coords_max[node_label, coord] = np.max(
                    orgnl_coords[:, node_label, coord])
                
                mean_val = np.mean(nrmlzd_orgnl_coords[:, node_label, coord])
                std_val = np.std(nrmlzd_orgnl_coords[:, node_label, coord])
                nrmlzd_orgnl_coords_min[node_label, coord] = np.min(
                    nrmlzd_orgnl_coords[:, node_label, coord])
                nrmlzd_orgnl_coords_mean_minus_std[node_label, coord] = (
                    mean_val - std_val
                )
                nrmlzd_orgnl_coords_mean[node_label, coord] = mean_val
                nrmlzd_orgnl_coords_mean_plus_std[node_label, coord] = (
                    mean_val + std_val
                )
                nrmlzd_orgnl_coords_max[node_label, coord] = np.max(
                    nrmlzd_orgnl_coords[:, node_label, coord])
        for node_label in range(n):
            for coord in range(dim):
                mean_val = np.mean(coords[:, node_label, coord])
                std_val = np.std(coords[:, node_label, coord])
                coords_min[node_label, coord] = np.min(
                    coords[:, node_label, coord])
                coords_mean_minus_std[node_label, coord] = (
                    mean_val - std_val
                )
                coords_mean[node_label, coord] = mean_val
                coords_mean_plus_std[node_label, coord] = (
                    mean_val + std_val
                )
                coords_max[node_label, coord] = np.max(
                    coords[:, node_label, coord])
                
                mean_val = np.mean(nrmlzd_coords[:, node_label, coord])
                std_val = np.std(nrmlzd_coords[:, node_label, coord])
                nrmlzd_coords_min[node_label, coord] = np.min(
                    nrmlzd_coords[:, node_label, coord])
                nrmlzd_coords_mean_minus_std[node_label, coord] = (
                    mean_val - std_val
                )
                nrmlzd_coords_mean[node_label, coord] = mean_val
                nrmlzd_coords_mean_plus_std[node_label, coord] = (
                    mean_val + std_val
                )
                nrmlzd_coords_max[node_label, coord] = np.max(
                    nrmlzd_coords[:, node_label, coord])
        
        # Plot results
        node_labels = np.arange(n, dtype=int)
        for coord in range(dim):
            plt_filename = filename + "-"
            plt.fill_between(
                node_labels, orgnl_coords_min[:, coord],
                orgnl_coords_max[:, coord], color="skyblue", alpha=0.25)
            plt.fill_between(
                node_labels, orgnl_coords_mean_minus_std[:, coord],
                orgnl_coords_mean_plus_std[:, coord], color="steelblue",
                alpha=0.25)
            plt.plot(
                node_labels, orgnl_coords_mean[:, coord], linestyle="-",
                color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            plt.ylim([0, L[coord]])
            if coord == 0:
                plt.ylabel("x", fontsize=16)
                plt_filename += "orgnl_x_vs_node_label"
            elif coord == 1:
                plt.ylabel("y", fontsize=16)
                plt_filename += "orgnl_y_vs_node_label"
            elif coord == 2:
                plt.ylabel("z", fontsize=16)
                plt_filename += "orgnl_z_vs_node_label"
            plt.title(
                "{}D auelp network random node placement".format(dim),
                fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            plt_filename += "_{}d_auelp_network_node_label_localization.png".format(dim)
            plt.savefig(plt_filename)
            plt.close()
        
        for coord in range(dim):
            plt_filename = filename + "-"
            plt.fill_between(
                node_labels, coords_min[:, coord], coords_max[:, coord],
                color="skyblue", alpha=0.25)
            plt.fill_between(
                node_labels, coords_mean_minus_std[:, coord],
                coords_mean_plus_std[:, coord], color="steelblue", alpha=0.25)
            plt.plot(
                node_labels, coords_mean[:, coord], linestyle="-",
                color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            plt.ylim([0, L[coord]])
            if coord == 0:
                plt.ylabel("x", fontsize=16)
                plt_filename += "x_vs_node_label"
            elif coord == 1:
                plt.ylabel("y", fontsize=16)
                plt_filename += "y_vs_node_label"
            elif coord == 2:
                plt.ylabel("z", fontsize=16)
                plt_filename += "z_vs_node_label"
            plt.title(
                "{}D auelp network Hilbert node labeling".format(dim),
                fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            plt_filename += "_{}d_auelp_network_node_label_localization.png".format(dim)
            plt.savefig(plt_filename)
            plt.close()

        for coord in range(dim):
            plt_filename = filename + "-"
            plt.fill_between(
                node_labels, nrmlzd_orgnl_coords_min[:, coord],
                nrmlzd_orgnl_coords_max[:, coord], color="skyblue", alpha=0.25)
            plt.fill_between(
                node_labels, nrmlzd_orgnl_coords_mean_minus_std[:, coord],
                nrmlzd_orgnl_coords_mean_plus_std[:, coord], color="steelblue",
                alpha=0.25)
            plt.plot(
                node_labels, nrmlzd_orgnl_coords_mean[:, coord], linestyle="-",
                color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            plt.ylim([0, 1])
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                plt_filename += "nrmlzd_orgnl_x_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                plt_filename += "nrmlzd_orgnl_y_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                plt_filename += "nrmlzd_orgnl_z_vs_node_label"
            plt.title(
                "{}D auelp network random node placement".format(dim),
                fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            plt_filename += "_{}d_auelp_network_node_label_localization.png".format(dim)
            plt.savefig(plt_filename)
            plt.close()
        
        for coord in range(dim):
            plt_filename = filename + "-"
            plt.fill_between(
                node_labels, nrmlzd_coords_min[:, coord],
                nrmlzd_coords_max[:, coord], color="skyblue", alpha=0.25)
            plt.fill_between(
                node_labels, nrmlzd_coords_mean_minus_std[:, coord],
                nrmlzd_coords_mean_plus_std[:, coord], color="steelblue",
                alpha=0.25)
            plt.plot(
                node_labels, nrmlzd_coords_mean[:, coord], linestyle="-",
                color="tab:blue")
            plt.xlabel("Node label", fontsize=16)
            plt.ylim([0, 1])
            if coord == 0:
                plt.ylabel("x/L", fontsize=16)
                plt_filename += "nrmlzd_x_vs_node_label"
            elif coord == 1:
                plt.ylabel("y/L", fontsize=16)
                plt_filename += "nrmlzd_y_vs_node_label"
            elif coord == 2:
                plt.ylabel("z/L", fontsize=16)
                plt_filename += "nrmlzd_z_vs_node_label"
            plt.title(
                "{}D auelp network Hilbert node labeling".format(dim),
                fontsize=16)
            plt.grid(True, alpha=0.25, zorder=0)
            plt.tight_layout()
            plt_filename += "_{}d_auelp_network_node_label_localization.png".format(dim)
            plt.savefig(plt_filename)
            plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial uniform end-linked polymer network topology augmentation plotting took {execution_time} seconds to run")