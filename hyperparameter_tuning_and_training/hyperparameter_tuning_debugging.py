# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import copy
import numpy as np
import torch
import torch.nn as nn
from hyperparameter_tuning_and_training.load_modules import (
    load_dataloaders,
    load_model,
    calculate_num_model_parameters,
    load_early_stopping
)
from src.file_io.file_io import (
    chkpnt_filepath_str,
    early_stop_filepath_str,
    chkpnt_filename_str,
    early_stop_filename_str
)
from src.helpers.model_utils import batch_analysis

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    torch.manual_seed(cfg.general.seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
        print("Cached:", round(torch.cuda.memory_reserved(0)/1024**3, 1), "GB")
    
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
    _ = chkpnt_filepath_str(aelp_network)
    _ = early_stop_filepath_str(aelp_network)

    # Load in the train and validation dataloaders
    train_dataloader, val_dataloader, _ = load_dataloaders(cfg)

    # Determine the number of nodes in each graph in the dataset
    data = next(iter(train_dataloader))
    batch, n, batch_size = batch_analysis(data)
    n_graph = n // batch_size
    assert n % batch_size == 0
    
    # Load in the model, optimizer, and starting epoch
    model, optimizer, start_epoch = load_model(cfg, n_graph, device)

    print(calculate_num_model_parameters(model))

if __name__ == "__main__":
    # import time
    
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()

    # execution_time = end_time - start_time
    # print(f"Hyperparameter tuning debugging code took {execution_time} seconds to run")