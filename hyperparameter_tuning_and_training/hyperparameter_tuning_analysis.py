# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import (
    chkpnt_filepath_str,
    early_stop_filepath_str,
    chkpnt_filename_str
)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    aelp_network = list(cfg.networks.keys())[0]
    if aelp_network not in ["abelp", "apelp", "auelp"]:
        error_str = (
            "The specified network must be either ``abelp'', "
            + "``apelp'', or ``auelp''."
        )
        print(error_str)
        return None
    synthesis_protocol_params = cfg.model.synthesis_protocol_params
    property_descriptors = cfg.model.property_descriptors
    if not (synthesis_protocol_params and property_descriptors):
        error_str = (
            "This analysis is tailored to the decoding of the synthesis "
            + "protocol parameters and the property descriptors."
        )
        print(error_str)
        return None
    _ = chkpnt_filepath_str(aelp_network)
    _ = early_stop_filepath_str(aelp_network)

    cfg_label_list = [
        "20250831A",
        "20250831B",
        "20250831C",
        "20250831D",
        "20250831E",
        "20250831F",
        "20250831G",
        "20250831H",
        "20250831I",
        "20250831J",
        "20250831K",
        "20250831L",
        "20250831M",
        "20250831N",
        "20250831O",
        "20250831P",
        "20250831Q",
        "20250831R"
    ]
    num_cfg_labels = len(cfg_label_list)
    cfg_early_stop_epochs = np.zeros(num_cfg_labels, dtype=int)
    cfg_total_eval_compnts = np.zeros((num_cfg_labels, 11))

    for cfg_indx, label in enumerate(cfg_label_list):
        total_eval_compnts_filename = (
            chkpnt_filename_str(aelp_network, label) + "-total_eval_compnts.dat"
        )
        total_eval_compnts = np.loadtxt(total_eval_compnts_filename)
        early_stop_epoch = np.where(np.any(total_eval_compnts!=0, axis=1))[0][-1]
        cfg_early_stop_epochs[cfg_indx] = early_stop_epoch
        cfg_total_eval_compnts[cfg_indx] = total_eval_compnts[early_stop_epoch]
    
    # Isolate evaluation components only
    cfg_total_eval_compnts = cfg_total_eval_compnts[:, 1:]
    
    # Apply min-max normalization for each evaluation component
    cfg_minmaxnorm_total_eval_compnts = np.zeros((num_cfg_labels, 10))
    for compnt_indx in range(10):
        total_eval_compnt = cfg_total_eval_compnts[:, compnt_indx]
        total_eval_compnt_min = np.min(total_eval_compnt)
        total_eval_compnt_max = np.max(total_eval_compnt)
        minmaxnorm_total_eval_compnt = (
            (total_eval_compnt-total_eval_compnt_min)
            / (total_eval_compnt_max-total_eval_compnt_min)
        )
        cfg_minmaxnorm_total_eval_compnts[:, compnt_indx] = (
            minmaxnorm_total_eval_compnt
        )

    # Calculate comprehensive evaluation score, i.e., L2-norm distance
    # in evaluation component hyperspace from origin
    ces = np.linalg.norm(cfg_minmaxnorm_total_eval_compnts, axis=1)

    # Print results
    print(cfg_early_stop_epochs)
    print(cfg_total_eval_compnts)
    print(cfg_minmaxnorm_total_eval_compnts)
    print(ces)
    print(cfg_label_list[np.argmin(ces)])

if __name__ == "__main__":
    # import time
    
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()

    # execution_time = end_time - start_time
    # print(f"Hyperparameter tuning analysis code took {execution_time} seconds to run")