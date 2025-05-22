# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import _config_filename_str
from src.networks.apelp_networks_config import sample_config_params_arr_func

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(cfg)
    # print("\n")
    # print(cfg.networks.apelp.label.network)
    # print(cfg.networks.apelp.label.date)
    # print(cfg.networks.apelp.label.batch)
    # print(cfg.networks.apelp.topology)

    # sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
    #     cfg)
    # raw_file_names_list = []

    # date = cfg.networks.apelp.label.date
    # batch = cfg.networks.apelp.label.batch
    # for indx in range(sample_config_num):
    #     sample = int(sample_config_params_arr[indx, 0])
    #     config = int(sample_config_params_arr[indx, 9])
    #     raw_file_names_list.append(
    #             _config_filename_str(date, batch, sample, config)+".npz")
    
    # raw_file_names_arr = np.asarray(raw_file_names_list)
    # print(type(raw_file_names_arr), type(raw_file_names_arr[0]), type(str(raw_file_names_arr[0])))
    # print(str(raw_file_names_arr[0]))

    print(cfg.general.normalize_nodal_coordinates)
    print(cfg.general.normalize_graph_descriptors)
    print(cfg.networks.apelp.topology.b)
    print(cfg.networks.apelp.topology.en)
    print(cfg.networks.apelp.topology.en[0])
    print(cfg.networks.apelp.topology.en[0][1])



    # print(cfg.networks)
    # print(cfg.networks.apelp.topology)
    # print(cfg.networks.apelp.label)
    # print(cfg.networks.apelp.synthesis)
    # dataset_config = cfg.dataset
    
    # if dataset_config.network not in [
    #     "abelp", "apelp", "auelp", "delaunay", "swidt", "voronoi"
    # ]:
    #     error_str = (
    #         "The specified network must be either ``abelp'', "
    #         + "``apelp'', ``auelp'', ``delaunay'', ``swidt'', "
    #         + "or ``voronoi''."
    #     )
    #     print(error_str)
    #     return None
    # else:
    #     if dataset_config.network == "abelp":
    #         pass
    #     elif dataset_config.network == "apelp":
    #         pass
    #     elif dataset_config.network == "auelp":
    #         pass
    #     elif dataset_config.network == "delaunay":
    #         pass
    #     elif dataset_config.network == "swidt":
    #         pass
    #     elif dataset_config.network == "voronoi":
    #         pass

if __name__ == "__main__":
    main()