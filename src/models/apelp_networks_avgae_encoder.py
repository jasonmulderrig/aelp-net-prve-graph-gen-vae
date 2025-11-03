import numpy as np
import torch
import torch.nn as nn
from src.models.apelp_networks_dataset import valid_params
from src.helpers.model_utils import (
    aelp_d_func,
    batch_analysis
)
from src.models.apelp_networks_avgae_modules import (
    apelpMultiedgeOrderTopologyEncoder,
    apelpFlatCoordsEncoder
)

class apelpAVGAEEncoder(nn.Module):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(self, cfg, n_graph):
        super().__init__()
        dim, _, _, _, _, _, self.multiedge_max = valid_params(cfg)
        assert dim == 3
        # Number of nodes in each individual pre-batched graph
        self.n_graph = n_graph

        # Extract latent space dimensions for the attributed network
        # embedding method, starting with the topology and nodal
        # coordinate latent space dimensions for the single-edge order
        # network, followed by the topology latent space dimensions for
        # higher-edge order networks
        self = aelp_d_func(
            self, cfg.model.d_adj, cfg.model.d_coords, cfg.model.d_adj_coords,
            self.multiedge_max)
        
        # Initialize ModuleList for topology encoder(s), and then gather
        # topology encoder for the single-edge and higher-edge order
        # networks
        self.adj_encoders_multiedge_order = nn.ModuleList()
        self.adj_encoders_multiedge_order.append(
            apelpMultiedgeOrderTopologyEncoder(
                cfg.model.node_dim, cfg.model.edge_dim, cfg.model.adj_hidden_dim,
                self.d_adj_combnd_0, cfg.model.p_dropout,
                cfg.model.supply_edge_dim, cfg.model.train_eps, cfg.model.inplace))
        for multiedge in range(1, self.multiedge_max):
            self.adj_encoders_multiedge_order.append(
                apelpMultiedgeOrderTopologyEncoder(
                    cfg.model.node_dim, cfg.model.edge_dim,
                    cfg.model.adj_hidden_dim,
                    getattr(self, f"d_adj_{multiedge:d}"), cfg.model.p_dropout,
                    cfg.model.supply_edge_dim, cfg.model.train_eps,
                    cfg.model.inplace))
        
        # Gather flat nodal coordinates encoder for the single-edge
        # order network
        self.flat_coords_encoder = apelpFlatCoordsEncoder(
            3*self.n_graph, cfg.model.coords_hidden_dims,
            self.d_coords_combnd_0*self.n_graph, cfg.model.p_dropout,
            cfg.model.inplace)

    def forward(self, data: torch.Tensor):
        # Extract the batch, the number of nodes from the as-provided
        # network for the data batch, the batch size, and the node
        # coordinates
        batch, n, batch_size = batch_analysis(data)
        assert self.n_graph == (n//batch_size)
        coords = data.coords

        # Topology encoder for the multiedge order networks
        mu_adj_multiedge_order = []
        logstd_adj_multiedge_order = []
        for multiedge in range(self.multiedge_max):
            mu_adj, logstd_adj = self.adj_encoders_multiedge_order[multiedge](
                getattr(data, f"x_{multiedge:d}"),
                getattr(data, f"edge_index_{multiedge:d}"),
                getattr(data, f"edge_attr_{multiedge:d}"))
            mu_adj_multiedge_order.append(mu_adj)
            logstd_adj_multiedge_order.append(logstd_adj)
        
        # Flat nodal coordinates encoder
        flat_coords_batch = (
            torch.zeros((batch_size, 3*self.n_graph)).to(batch.device)
        )
        for batch_indx in range(batch_size):
            # [self.n_graph, 3]
            coords_graph = (
                coords[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :]
            )
            # [3*self.n_graph]
            flat_coords_batch[batch_indx] = torch.hstack(
                (coords_graph[:, 0], coords_graph[:, 1], coords_graph[:, 2]))
        # [batch_size, 3*self.n_graph] -> [batch_size, self.d_coords_combnd_0*self.n_graph]
        mu_flat_coords_batch, logstd_flat_coords_batch = (
            self.flat_coords_encoder(flat_coords_batch)
        )
        mu_coords = torch.zeros((n, self.d_coords_combnd_0)).to(batch.device)
        logstd_coords = torch.zeros((n, self.d_coords_combnd_0)).to(batch.device)
        for batch_indx in range(batch_size):
            # [self.d_coords_combnd_0*self.n_graph] -> [self.n_graph, self.d_coords_combnd_0]
            mu_coords[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :] = (
                mu_flat_coords_batch[batch_indx].reshape((self.n_graph, self.d_coords_combnd_0))
            )
            logstd_coords[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :] = (
                logstd_flat_coords_batch[batch_indx].reshape((self.n_graph, self.d_coords_combnd_0))
            )

        # Initialize the latent space for the multiedge order network
        mu = torch.zeros((n, self.d)).to(batch.device)
        logstd = torch.zeros((n, self.d)).to(batch.device)

        # Assemble the latent space for the single-edge order network
        # via the attributed network embedding method
        mu[:, :self.d_adj_0] = mu_adj_multiedge_order[0][:, :self.d_adj_0]
        mu[:, self.d_adj_0:self.d_adj_combnd_0] = (
            0.5 * (
                mu_adj_multiedge_order[0][:, self.d_adj_0:]
                + mu_coords[:, :self.d_adj_coords_0]
            )
        )
        mu[:, self.d_adj_combnd_0:self.d_0] = mu_coords[:, self.d_adj_coords_0:]

        logstd[:, :self.d_adj_0] = (
            logstd_adj_multiedge_order[0][:, :self.d_adj_0]
        )
        logstd[:, self.d_adj_0:self.d_adj_combnd_0] = (
            0.5 * (
                logstd_adj_multiedge_order[0][:, self.d_adj_0:]
                + logstd_coords[:, :self.d_adj_coords_0]
            )
        )
        logstd[:, self.d_adj_combnd_0:self.d_0] = (
            logstd_coords[:, self.d_adj_coords_0:]
        )

        # Assemble the latent space for the higher-edge order networks
        # via the attributed network embedding method
        for multiedge in range(1, self.multiedge_max):
            mu[:, getattr(self, f"d_{(multiedge-1):d}"):getattr(self, f"d_{multiedge:d}")] = (
                mu_adj_multiedge_order[multiedge]
            )
            logstd[:, getattr(self, f"d_{(multiedge-1):d}"):getattr(self, f"d_{multiedge:d}")] = (
                logstd_adj_multiedge_order[multiedge]
            )

        return mu, logstd