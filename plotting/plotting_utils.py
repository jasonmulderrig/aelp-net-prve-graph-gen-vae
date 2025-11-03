import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "cm"

def dim_2_network_topology_axes_formatter(
        ax: plt.axes,
        core_square: np.ndarray,
        core_square_color: str,
        core_square_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    ax.plot(
        core_square[:, 0], core_square[:, 1],
        color=core_square_color, linewidth=core_square_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax

def dim_3_network_topology_axes_formatter(
        ax: plt.axes,
        core_cube: np.ndarray,
        core_cube_color: str,
        core_cube_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        zlim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        zticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    for face in np.arange(6):
        ax.plot(
            core_cube[face, :, 0], core_cube[face, :, 1], core_cube[face, :, 2],
            color=core_cube_color, linewidth=core_cube_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax

def data_histogram_plotter(
        data: list[int] | list[float] | np.ndarray,
        bins: str | list[int] | list[float] | np.ndarray,
        xlabel: str,
        filename: str) -> None:
    data = np.asarray(data)
    data = data[~np.isnan(data)]

    x_min = np.min(data) - 0.1 * np.ptp(data)
    x_min = np.max(np.asarray([x_min, 0]))
    x_max = np.max(data) + 0.1 * np.ptp(data)
    x_max = np.min(np.asarray([x_max, 1]))
    
    fig, ax = plt.subplots()

    counts, bins = np.histogram(data, bins=bins, density=True)
    ax.hist(
        bins[:-1], bins, weights=counts, color="tab:blue", edgecolor="black",
        linewidth=0.25, zorder=3)
    # Fill in additional plot formatting later!!!
    # ax.set_xlim((x_min, x_max))
    # ax.set_ylim(ylim)
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title, fontsize=20)
    ax.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def data_bihistogram_plotter(
        top_data: list[int] | list[float] | np.ndarray,
        bottom_data: list[int] | list[float] | np.ndarray,
        top_data_bins: str | list[int] | list[float] | np.ndarray,
        bottom_data_bins: str | list[int] | list[float] | np.ndarray,
        xlabel: str,
        filename: str) -> None:
    # top_bins and bottom_bins?
    top_data = np.asarray(top_data)
    top_data = top_data[~np.isnan(top_data)]
    
    bottom_data = np.asarray(bottom_data)
    bottom_data = bottom_data[~np.isnan(bottom_data)]

    top_x_min = np.min(top_data) - 0.1 * np.ptp(top_data)
    top_x_min = np.max(np.asarray([top_x_min, 0]))
    top_x_max = np.max(top_data) + 0.1 * np.ptp(top_data)
    top_x_max = np.min(np.asarray([top_x_max, 1]))
    
    bottom_x_min = np.min(bottom_data) - 0.1 * np.ptp(bottom_data)
    bottom_x_min = np.max(np.asarray([bottom_x_min, 0]))
    bottom_x_max = np.max(bottom_data) + 0.1 * np.ptp(bottom_data)
    bottom_x_max = np.min(np.asarray([bottom_x_max, 1]))

    fig, ax = plt.subplots()

    top_data_counts, top_data_bins = np.histogram(
        top_data, bins=top_data_bins, density=True)
    bottom_data_counts, bottom_data_bins = np.histogram(
        bottom_data, bins=bottom_data_bins, density=True)
    ax.hist(
        top_data_bins[:-1], top_data_bins,
        weights=top_data_counts, color="tab:blue", edgecolor="black",
        linewidth=0.25, zorder=3)
    ax.hist(
        bottom_data_bins[:-1], bottom_data_bins,
        weights=-bottom_data_counts, color="tab:olive", edgecolor="black",
        linewidth=0.25, zorder=3)
    ax.axhline(0, color="black", linewidth=1)
    # Fill in additional plot formatting later!!!
    # ax.set_xlim((x_min, x_max))
    # ax.set_ylim(ylim)
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    ax.set_xlabel(xlabel, fontsize=12)
    # ax.set_ylabel(ylabel)
    # ax.set_title(title, fontsize=20)
    ax.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()