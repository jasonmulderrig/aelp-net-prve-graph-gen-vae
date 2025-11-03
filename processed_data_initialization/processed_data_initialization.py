# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import torch

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    aelp_network = list(cfg.networks.keys())[0]
    if aelp_network not in ["abelp", "apelp", "auelp"]:
        error_str = (
            "The specified network must be either ``abelp'', "
            + "``apelp'', or ``auelp''."
        )
        print(error_str)
        return None
    else:
        # Additional loading for GGML models once that code is complete!!!
        if aelp_network == "abelp":
            from src.models.abelp_networks_dataset import abelpDataset
            aelpDataset = abelpDataset
        elif aelp_network == "apelp":
            from src.models.apelp_networks_dataset import apelpDataset
            aelpDataset = apelpDataset
        elif aelp_network == "auelp":
            from src.models.auelp_networks_dataset import auelpDataset
            aelpDataset = auelpDataset

    train_dataset = aelpDataset(cfg, "train")
    val_dataset = aelpDataset(cfg, "val")
    test_dataset = aelpDataset(cfg, "test")

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Processed data initialization took {execution_time} seconds to run")