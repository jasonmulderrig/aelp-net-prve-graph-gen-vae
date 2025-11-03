import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    log_loss
)

def aelp_d_func(
    obj, d_adj: int, d_coords: int, d_adj_coords: int, multiedge_max: int):
    # Extract latent space dimensions for the attributed network
    # embedding method, starting with the topology and nodal coordinate
    # latent space dimensions for the single-edge order network,
    # followed by the topology latent space dimensions for higher-edge
    # order networks
    obj.d_adj_0 = d_adj
    obj.d_coords_0 = d_coords
    obj.d_adj_coords_0 = d_adj_coords
    obj.d_adj_combnd_0 = obj.d_adj_0 + obj.d_adj_coords_0
    obj.d_coords_combnd_0 = obj.d_coords_0 + obj.d_adj_coords_0
    d = obj.d_adj_0 + obj.d_adj_coords_0 + obj.d_coords_0
    obj.d_0 = d
    for multiedge in range(1, multiedge_max):
        d_adj_multiedge = d_adj
        d += d_adj_multiedge
        setattr(obj, f"d_adj_{multiedge:d}", d_adj_multiedge)
        setattr(obj, f"d_{multiedge:d}", d)
    obj.d = d
    return obj

def batch_analysis(data):
    batch = data.batch
    n = data.num_nodes
    batch_size = torch.max(batch).item() + 1
    return batch, n, batch_size

def cyclic_beta_kl_div_annealing(
    epoch: int, n_epochs_per_anneal_cycle: int,
    rampup_anneal_cycle_ratio: float, max_beta: float) -> float:
    progress = (epoch%n_epochs_per_anneal_cycle) / n_epochs_per_anneal_cycle
    if progress < rampup_anneal_cycle_ratio:
        return max_beta * progress / rampup_anneal_cycle_ratio
    else: return max_beta

def apply_threshold(adj, threshold_val: float):
    adj = (adj>threshold_val)
    return adj

def decoded_z_to_adj_or_adj_logits(
    z_adj: torch.Tensor, sigmoid: bool, bernoulli: bool, symtry: bool,
    threshold: bool, threshold_val: float):
    adj_logits = torch.matmul(z_adj, z_adj.t())
    if sigmoid:
        adj = torch.sigmoid(adj_logits)
        if bernoulli: adj = torch.bernoulli(adj)
        if symtry: adj = torch.maximum(adj, adj.t())
        if threshold:
            if not bernoulli: adj = apply_threshold(adj, threshold_val)
        return adj
    else:
        if symtry:
            adj_logits = torch.where(
                adj_logits.abs()>=adj_logits.abs().t(), adj_logits,
                adj_logits.t())
        return adj_logits

def triu_adj_edges(adj: torch.Tensor):
    return adj[torch.triu(torch.ones_like(adj))==1]

def eval_func(y_true: np.ndarray, y_pred: np.ndarray, func_str: str) -> float:
    if func_str == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif func_str == "mse":
        return mean_squared_error(y_true, y_pred)
    elif func_str == "r2":
        return 1. - r2_score(y_true, y_pred)
    elif func_str == "ap":
        return 1. - average_precision_score(y_true, y_pred)
    elif func_str == "gini":
        return 1. - (2.*roc_auc_score(y_true, y_pred)-1.)
    elif func_str == "ba":
        return 1. - balanced_accuracy_score(y_true, y_pred)
    elif func_str == "f1":
        return 1. - f1_score(y_true, y_pred, zero_division=0.0)
    elif func_str == "log_loss":
        return log_loss(y_true, y_pred)
    else:
        error_str = "The called-for evaluation function is not implemented!"
        raise ValueError(error_str)