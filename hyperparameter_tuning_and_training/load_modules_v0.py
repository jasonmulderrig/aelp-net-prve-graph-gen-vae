# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
from src.file_io.file_io import (
    chkpnt_filename_str,
    early_stop_filename_str
)

def load_dataloaders(cfg):
    aelp_network = list(cfg.networks.keys())[0]
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

    return (
        DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    )

def load_model(cfg, device):
    aelp_network = list(cfg.networks.keys())[0]
    if aelp_network == "abelp":
        from src.models.abelp_networks_avgae import abelpAVGAE
        model = abelpAVGAE(cfg).to(device)
    elif aelp_network == "apelp":
        from src.models.apelp_networks_avgae import apelpAVGAE
        model = apelpAVGAE(cfg).to(device)
    elif aelp_network == "auelp":
        from src.models.auelp_networks_avgae import auelpAVGAE
        model = auelpAVGAE(cfg).to(device)
    
    for params in model.parameters():
        if params.dim() == 1: nn.init.constant_(params, 0)
        else: nn.init.xavier_normal_(params)
    
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    start_epoch = 0
    
    if os.path.exists(chkpnt_filename_str(aelp_network, cfg.label)+".model"):
        chkpnt = torch.load(
            chkpnt_filename_str(aelp_network, cfg.label)+".model",
            weights_only=True)
        model.load_state_dict(chkpnt["model_state_dict"])
        optimizer.load_state_dict(chkpnt["optimizer_state_dict"])
        start_epoch = chkpnt["epoch"] + 1
    
    return model, optimizer, start_epoch

def calculate_num_model_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_params, num_trainable_params

def load_early_stopping(cfg):
    aelp_network = list(cfg.networks.keys())[0]
    best_model_state_dict = None
    min_total_eval_metric = 1e10
    min_total_eval_metric_epoch = 0
    patience_counter = 0
    start_epoch_early_stop = 0
    
    if os.path.exists(early_stop_filename_str(aelp_network, cfg.label)+".model"):
        early_stop_dict = torch.load(
            early_stop_filename_str(aelp_network, cfg.label)+".model",
            weights_only=True)
        best_model_state_dict = early_stop_dict["best_model_state_dict"]
        min_total_eval_metric = early_stop_dict["min_total_eval_metric"]
        min_total_eval_metric_epoch = (
            early_stop_dict["min_total_eval_metric_epoch"]
        )
        patience_counter = early_stop_dict["patience_counter"]
        start_epoch_early_stop = early_stop_dict["epoch"] + 1
    
    return (
        best_model_state_dict, min_total_eval_metric,
        min_total_eval_metric_epoch, patience_counter, start_epoch_early_stop
    )