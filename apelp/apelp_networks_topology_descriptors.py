# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import multiprocessing
import random
from src.helpers.multiprocessing_utils import (
    run_aelp_network_local_topological_descriptor,
    run_aelp_network_local_multiedge_order_topological_descriptor,
    run_aelp_network_global_property_descriptor
)
from src.networks.apelp_networks_config import (
    params_list_func,
    sample_config_params_arr_func
)

@hydra.main(
        version_base=None,
        config_path="../configs/networks/apelp",
        config_name="apelp_networks")
def main(cfg: DictConfig) -> None:
    # Gather arrays of configuration parameters
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)

    ##### Calculate descriptors
    print("Calculating descriptors", flush=True)

    local_topological_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2]]
    ) # sample, config, b
    local_topological_descriptors_params_list = params_list_func(
        local_topological_descriptors_params_arr)
    local_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *lcl_tplgcl_dscrptrs)
            for (sample, config, b) in local_topological_descriptors_params_list
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_topological_descriptor_args)

    local_multiedge_order_topological_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2, 6]]
    ) # sample, config, b, k
    local_multiedge_order_topological_descriptors_params_list = params_list_func(
        local_multiedge_order_topological_descriptors_params_arr)
    local_multiedge_order_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, int(k), *lcl_tplgcl_dscrptrs)
            for (sample, config, b, k) in local_multiedge_order_topological_descriptors_params_list
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_multiedge_order_topological_descriptor_args)

    global_property_descriptors_params_arr = (
        sample_config_params_arr[:, [0, 10, 2]]
    ) # sample, config, b
    global_property_descriptors_params_list = params_list_func(
        global_property_descriptors_params_arr)
    global_property_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_prprty_dscrptrs)
            for (sample, config, b) in global_property_descriptors_params_list
            for glbl_prprty_dscrptrs in list(map(tuple, cfg.descriptors.global_property_descriptors))
        ]
    )
    random.shuffle(global_property_descriptor_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_aelp_network_local_topological_descriptor,
            local_topological_descriptor_args)
        pool.map(
            run_aelp_network_local_multiedge_order_topological_descriptor,
            local_multiedge_order_topological_descriptor_args)
        pool.map(
            run_aelp_network_global_property_descriptor,
            global_property_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network topology descriptors calculation took {execution_time} seconds to run")