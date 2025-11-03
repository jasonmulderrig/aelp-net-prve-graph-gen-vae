# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import filepath_str
from plotting.plotting_utils import data_histogram_plotter

class Namespace(): pass

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    filepath = filepath_str("analysis")
    sample_configs_mu_filename = filepath + "sample_configs_mu" + ".npy"
    sample_configs_logstd_filename = filepath + "sample_configs_logstd" + ".npy"

    sample_configs_mu = np.load(sample_configs_mu_filename)
    sample_configs_logstd = np.load(sample_configs_logstd_filename)

    sample_configs_mu = sample_configs_mu.flatten()
    sample_configs_logstd = sample_configs_logstd.flatten()

    sample_configs_mu_plt_filename = filepath + "sample_configs_mu" + ".png"
    sample_configs_logstd_plt_filename = filepath + "sample_configs_logstd" + ".png"

    data_histogram_plotter(
        sample_configs_mu,
        np.linspace(np.min(sample_configs_mu), np.max(sample_configs_mu), 16),
        "$\\mu$", sample_configs_mu_plt_filename)
    data_histogram_plotter(
        sample_configs_logstd,
        np.linspace(np.min(sample_configs_logstd), np.max(sample_configs_logstd), 16),
        "$\\log(\\sigma)$", sample_configs_logstd_plt_filename)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Trained latent space parameter plotting took {execution_time} seconds to run")