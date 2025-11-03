import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GINEConv, GINConv, BatchNorm

def graph_conv_mlp(
    input_dim: int, hidden_dim: int, p_dropout: float, inplace: bool):
    return (
        nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=inplace),
            nn.Dropout(p=p_dropout, inplace=inplace),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=inplace),
            nn.Dropout(p=p_dropout, inplace=inplace),
            nn.Linear(hidden_dim, hidden_dim))
    )

def decoder_mlp(
    input_dim: int, hidden_dims: list[int], output_dim: int,
    final_activation: str, p_dropout: float, batchnorm: bool, inplace: bool):
    mlp_module_list = nn.ModuleList()
    mlp_module_list.append(nn.Linear(input_dim, hidden_dims[0]))
    if batchnorm: mlp_module_list.append(nn.BatchNorm1d(hidden_dims[0]))
    mlp_module_list.append(nn.ReLU(inplace=inplace))
    mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
    for layer in range(1, len(hidden_dims)-1):
        mlp_module_list.append(
            nn.Linear(hidden_dims[layer-1], hidden_dims[layer]))
        if batchnorm: mlp_module_list.append(nn.BatchNorm1d(hidden_dims[layer]))
        mlp_module_list.append(nn.ReLU(inplace=inplace))
        mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
    mlp_module_list.append(nn.Linear(hidden_dims[-2], hidden_dims[-1]))
    mlp_module_list.append(nn.ReLU(inplace=inplace))
    mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
    mlp_module_list.append(nn.Linear(hidden_dims[-1], output_dim))
    if final_activation == "softplus": mlp_module_list.append(nn.Softplus())
    elif final_activation == "sigmoid": mlp_module_list.append(nn.Sigmoid())
    return nn.Sequential(*mlp_module_list)

class apelpMultiedgeOrderTopologyEncoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, node_dim: int, edge_dim: int, hidden_dim: int, output_dim: int,
        p_dropout: float, supply_edge_dim: bool, train_eps: bool, inplace: bool):
        super().__init__()
        self.hidden_dim = hidden_dim
        # MLP linear layers
        gineconv_mlp_0_adj = graph_conv_mlp(
            node_dim, hidden_dim, p_dropout, inplace)
        gineconv_mlp_1_adj = graph_conv_mlp(
            hidden_dim, hidden_dim, p_dropout, inplace)

        # GINEConv layers
        if supply_edge_dim:
            self.gineconv_model_adj = Sequential("x, edge_index, edge_attr", [
                (GINEConv(gineconv_mlp_0_adj, edge_dim=edge_dim, train_eps=train_eps), "x, edge_index, edge_attr -> x"),
                (BatchNorm(hidden_dim), "x -> x"),
                nn.ReLU(inplace=inplace),
                nn.Dropout(p=p_dropout, inplace=inplace),
                (GINEConv(gineconv_mlp_1_adj, edge_dim=edge_dim, train_eps=train_eps), "x, edge_index, edge_attr -> x"),
                (BatchNorm(hidden_dim), "x -> x"),
                nn.ReLU(inplace=inplace),
                nn.Dropout(p=p_dropout, inplace=inplace)
            ])
        else:
            self.gineconv_model_adj = Sequential("x, edge_index, edge_attr", [
                (GINEConv(gineconv_mlp_0_adj, train_eps=train_eps), "x, edge_index, edge_attr -> x"),
                (BatchNorm(hidden_dim), "x -> x"),
                nn.ReLU(inplace=inplace),
                nn.Dropout(p=p_dropout, inplace=inplace),
                (GINEConv(gineconv_mlp_1_adj, train_eps=train_eps), "x, edge_index, edge_attr -> x"),
                (BatchNorm(hidden_dim), "x -> x"),
                nn.ReLU(inplace=inplace),
                nn.Dropout(p=p_dropout, inplace=inplace)
            ])
        
        # Linear layers accounting for mean and log standard deviation
        self.lin_mu_adj = nn.Linear(hidden_dim, output_dim)
        self.lin_logstd_adj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_attr: torch.Tensor):
        if edge_index.numel() > 0:
            x = self.gineconv_model_adj(x, edge_index, edge_attr)
        else: x = x.new_zeros((x.size(0), self.hidden_dim))
        return self.lin_mu_adj(x), self.lin_logstd_adj(x)

class apelpFlatCoordsEncoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int,
        p_dropout: float, inplace: bool):
        super().__init__()
        # Linear layers
        mlp_module_list = nn.ModuleList()
        mlp_module_list.append(nn.Linear(input_dim, hidden_dims[0]))
        mlp_module_list.append(nn.BatchNorm1d(hidden_dims[0]))
        mlp_module_list.append(nn.ReLU(inplace=inplace))
        mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
        for layer in range(1, len(hidden_dims)):
            mlp_module_list.append(
                nn.Linear(hidden_dims[layer-1], hidden_dims[layer]))
            mlp_module_list.append(nn.BatchNorm1d(hidden_dims[layer]))
            mlp_module_list.append(nn.ReLU(inplace=inplace))
            mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
        self.mlp_flat_coords = nn.Sequential(*mlp_module_list)

        # Linear layers accounting for mean and log standard deviation
        self.lin_mu_flat_coords = nn.Linear(hidden_dims[-1], output_dim)
        self.lin_logstd_flat_coords = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, flat_coords: torch.Tensor):
        flat_coords = self.mlp_flat_coords(flat_coords)
        return (
            self.lin_mu_flat_coords(flat_coords),
            self.lin_logstd_flat_coords(flat_coords)
        )

class apelpSpatialNetworkTopologyDecoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, feature: str, input_dim: int, hidden_dims: list[int],
        output_dim: int, p_dropout: float, inplace: bool):
        super().__init__()
        self.feature = feature
        if self.feature not in ["L", "flat_coords"]:
            error_str = (
                "The spatial network topology features that can be "
                + "decoded by this decoder module are either the "
                + "simulation box side length, L, or the flattened "
                + "nodal coordinates, flat_coords."
            )
            raise ValueError(error_str)
        if len(hidden_dims) < 3:
            error_str = (
                "There should be at least 3 hidden dimensions in the "
                + "hidden_dims list."
            )
            raise ValueError(error_str)
        if self.feature == "L" and output_dim != 1:
            error_str = (
                "output_dim = 1 must be true when decoding the "
                + "simulation box side length."
            )
            raise ValueError(error_str)
        
        # Linear layers
        if self.feature == "L":
            self.mlp = decoder_mlp(
                input_dim, hidden_dims, output_dim, "softplus", p_dropout, False,
                inplace)
        elif self.feature == "flat_coords":
            self.mlp = decoder_mlp(
                input_dim, hidden_dims, output_dim, "sigmoid", p_dropout, False,
                inplace)
    
    def forward(self, z: torch.Tensor):
        if self.feature == "L":
            return self.mlp(z.mean(dim=0).unsqueeze(0)).squeeze()
        elif self.feature == "flat_coords":
            return self.mlp(z.unsqueeze(0)).squeeze()

class apelpEdgeAttributeDecoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, edge_attribute: str, dim: int, hidden_dim: int, output_dim: int,
        p_dropout: float, train_eps: bool, inplace: bool):
        super().__init__()
        self.edge_attribute = edge_attribute
        if self.edge_attribute not in ["edge_type", "edge_l_cntr"]:
            error_str = (
                "The edge attributes that can be decoded by this "
                + "decoder module are either the edge type, edge_type, "
                + "or the edge contour length, edge_l_cntr."
            )
            raise ValueError(error_str)
        if output_dim != 1:
            error_str = "output_dim = 1 must be true."
            raise ValueError(error_str)
        
        # MLP linear layers
        ginconv_mlp_0_edge_attr = graph_conv_mlp(
            dim, hidden_dim, p_dropout, inplace)
        ginconv_mlp_1_edge_attr = graph_conv_mlp(
            hidden_dim, hidden_dim, p_dropout, inplace)
        
        # GINConv layers
        self.ginconv_model_edge_attr = Sequential("coords, edge_index", [
            (GINConv(ginconv_mlp_0_edge_attr, train_eps=train_eps), "coords, edge_index -> x"),
            (BatchNorm(hidden_dim), "x -> x"),
            nn.ReLU(inplace=inplace),
            nn.Dropout(p=p_dropout, inplace=inplace),
            (GINConv(ginconv_mlp_1_edge_attr, train_eps=train_eps), "x, edge_index -> x"),
            (BatchNorm(hidden_dim), "x -> x"),
            nn.ReLU(inplace=inplace),
            nn.Dropout(p=p_dropout, inplace=inplace)
        ])

        # Linear layers for decoding edge attribute
        mlp_module_list = nn.ModuleList()
        if self.edge_attribute == "edge_type":
            mlp_module_list.append(nn.Linear(hidden_dim*2, hidden_dim))
        elif self.edge_attribute == "edge_l_cntr":
            mlp_module_list.append(nn.Linear((hidden_dim*2)+1, hidden_dim))
        mlp_module_list.append(nn.BatchNorm1d(hidden_dim))
        mlp_module_list.append(nn.ReLU(inplace=inplace))
        mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
        mlp_module_list.append(nn.Linear(hidden_dim, hidden_dim//2))
        mlp_module_list.append(nn.ReLU(inplace=inplace))
        mlp_module_list.append(nn.Dropout(p=p_dropout, inplace=inplace))
        mlp_module_list.append(nn.Linear(hidden_dim//2, output_dim))
        if self.edge_attribute == "edge_l_cntr":
            mlp_module_list.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*mlp_module_list)
    
    def forward(self, coords: torch.Tensor, edge_index: torch.Tensor):
        if edge_index.numel() == 0:
            error_str = (
                "Edges must be supplied to the edge attribute decoder!"
            )
            raise ValueError(error_str)
        node_0, node_1 = edge_index
        x = self.ginconv_model_edge_attr(coords, edge_index)
        if self.edge_attribute == "edge_type":
            return self.mlp(torch.cat((x[node_0], x[node_1]), dim=-1)).squeeze(-1)
        elif self.edge_attribute == "edge_l_cntr":
            l = torch.norm(coords[node_1]-coords[node_0], dim=1, keepdim=True)
            return (
                self.mlp(torch.cat((x[node_0], x[node_1], l), dim=-1)).squeeze(-1)
            )

class apelpSynthesisProtocolParameterDecoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, parameter: str, input_dim: int, hidden_dims: list[int],
        output_dim: int, p_dropout: float, inplace: bool):
        super().__init__()
        self.parameter = parameter
        if self.parameter not in ["xi", "chi", "en_mean"]:
            error_str = (
                "The synthesis protocol parameters that can be decoded "
                + "by this decoder module are either the "
                + "chain-to-cross-link connection probability, xi, the "
                + "stoichiometric imbalance between cross-linker sites "
                + "and chain ends, chi, and the mean chain segment "
                + "particle number, en_mean."
            )
            raise ValueError(error_str)
        if len(hidden_dims) < 3:
            error_str = (
                "There should be at least 3 hidden dimensions in the "
                + "hidden_dims list."
            )
            raise ValueError(error_str)
        if output_dim != 1:
            error_str = "output_dim = 1 must be true"
            raise ValueError(error_str)
        
        # Linear layers
        if self.parameter in ["xi", "chi"]:
            self.mlp = decoder_mlp(
                input_dim, hidden_dims, output_dim, "softplus", p_dropout, False,
                inplace)
        elif self.parameter == "en_mean":
            self.mlp = decoder_mlp(
                input_dim, hidden_dims, output_dim, "sigmoid", p_dropout, False,
                inplace)
    
    def forward(self, z: torch.Tensor):
        return self.mlp(z.mean(dim=0).unsqueeze(0)).squeeze()

class apelpPropertyDescriptorDecoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(
        self, descriptor: str, input_dim: int, hidden_dims: list[int],
        output_dim: int, p_dropout: float, inplace: bool):
        super().__init__()
        self.descriptor = descriptor
        if self.descriptor not in ["eeel_dobrynin_kappa", "eeel_glbl_mean_gamma"]:
            error_str = (
                "The property descriptors that can be decoded by this "
                + "decoder module are either the elastically-effective "
                + "end-linked quality factor, eeel_dobrynin_kappa, or "
                + "the elastically-effective end-linked global mean "
                + "chain/edge stretch, eeel_glbl_mean_gamma."
            )
            raise ValueError(error_str)
        if len(hidden_dims) < 3:
            error_str = (
                "There should be at least 3 hidden dimensions in the "
                + "hidden_dims list."
            )
            raise ValueError(error_str)
        if output_dim != 1:
            error_str = "output_dim = 1 must be true"
            raise ValueError(error_str)
        
        # Linear layers
        self.mlp = decoder_mlp(
            input_dim, hidden_dims, output_dim, "sigmoid", p_dropout, False,
            inplace)
    
    def forward(self, z: torch.Tensor):
        return self.mlp(z.mean(dim=0).unsqueeze(0)).squeeze()