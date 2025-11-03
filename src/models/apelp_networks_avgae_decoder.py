import numpy as np
import torch
import torch.nn as nn
from src.models.apelp_networks_dataset import valid_params
from src.helpers.model_utils import (
    aelp_d_func,
    batch_analysis,
    decoded_z_to_adj_or_adj_logits,
    triu_adj_edges
)
from src.models.apelp_networks_avgae_modules import (
    apelpSpatialNetworkTopologyDecoder,
    apelpEdgeAttributeDecoder,
    apelpSynthesisProtocolParameterDecoder,
    apelpPropertyDescriptorDecoder
)
from torch_geometric.utils import (
    batched_negative_sampling,
    negative_sampling
)

class apelpAVGAEDecoder(nn.Module):
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
        
        # Indicators for adding functionality for predicting synthesis
        # protocol parameters and property descriptors
        self.synthesis_protocol_params = cfg.model.synthesis_protocol_params
        self.property_descriptors = cfg.model.property_descriptors
        
        # Initialize the simulation box side length decoder
        self.L_decoder = apelpSpatialNetworkTopologyDecoder(
            "L", self.d_coords_combnd_0, cfg.model.L_hidden_dims, 1,
            cfg.model.p_dropout, cfg.model.inplace)
        
        # Initialize the flat nodal coordinates decoder, and reverse the
        # order of the hidden dimensions used for the linear layers in
        # the corresponding encoder block
        self.flat_coords_decoder = apelpSpatialNetworkTopologyDecoder(
            "flat_coords", self.d_coords_combnd_0*self.n_graph,
            cfg.model.coords_hidden_dims[::-1], 3*self.n_graph,
            cfg.model.p_dropout, cfg.model.inplace)

        # Initialize the edge type and the edge contour length decoders
        self.edge_type_decoder = apelpEdgeAttributeDecoder(
            "edge_type", 3, cfg.model.edge_attr_hidden_dim, 1,
            cfg.model.p_dropout, cfg.model.train_eps, cfg.model.inplace)
        self.edge_l_cntr_decoder = apelpEdgeAttributeDecoder(
            "edge_l_cntr", 3, cfg.model.edge_attr_hidden_dim, 1,
            cfg.model.p_dropout, cfg.model.train_eps, cfg.model.inplace)
        
        # Initialize the synthesis protocol parameter decoders, if
        # called for
        if self.synthesis_protocol_params:
            self.xi_decoder = apelpSynthesisProtocolParameterDecoder(
                "xi", self.d, cfg.model.synthesis_protocol_params_hidden_dims,
                1, cfg.model.p_dropout, cfg.model.inplace)
            self.chi_decoder = apelpSynthesisProtocolParameterDecoder(
                "chi", self.d, cfg.model.synthesis_protocol_params_hidden_dims,
                1, cfg.model.p_dropout, cfg.model.inplace)
            self.en_mean_decoder = apelpSynthesisProtocolParameterDecoder(
                "en_mean", self.d,
                cfg.model.synthesis_protocol_params_hidden_dims,
                1, cfg.model.p_dropout, cfg.model.inplace)
        
        # Initialize the property descriptor decoders, if called for
        if self.property_descriptors:
            self.eeel_dobrynin_kappa_decoder = apelpPropertyDescriptorDecoder(
                "eeel_dobrynin_kappa", self.d,
                cfg.model.property_descriptors_hidden_dims,
                1, cfg.model.p_dropout, cfg.model.inplace)
            self.eeel_glbl_mean_gamma_decoder = apelpPropertyDescriptorDecoder(
                "eeel_glbl_mean_gamma", self.d,
                cfg.model.property_descriptors_hidden_dims,
                1, cfg.model.p_dropout, cfg.model.inplace)

    def forward(self, z: torch.Tensor, data: torch.Tensor):
        # Extract the batch, the number of nodes from the as-provided
        # network for the data batch, the batch size, and the number of 
        # negative edges to sample for each individual pre-batched graph
        # in the data batch
        batch, n, batch_size = batch_analysis(data)
        assert self.n_graph == (n//batch_size)

        # Extract the region of the latent space attributed to the nodal
        # coordinates for each graph in the data batch, and decode the
        # simulation box side length and nodal coordinates
        L = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
        coords = (
            torch.zeros((batch_size*self.n_graph, 3), dtype=torch.float).to(batch.device)
        )
        for batch_indx in range(batch_size):
            # [self.n_graph, self.d_coords_combnd_0]
            z_coords_graph = (
                z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), self.d_adj_0:self.d_0]
            )
            L[batch_indx] = self.L_decoder(z_coords_graph)
            # [self.n_graph, self.d_coords_combnd_0] -> [self.d_coords_combnd_0*self.n_graph] -> [3*self.n_graph]
            flat_coords_graph = self.flat_coords_decoder(z_coords_graph.flatten())
            x_coords_graph = flat_coords_graph[:self.n_graph]
            y_coords_graph = flat_coords_graph[self.n_graph:(2*self.n_graph)]
            z_coords_graph = flat_coords_graph[(2*self.n_graph):]
            # [3*self.n_graph] -> [self.n_graph, 3]
            coords[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :] = (
                torch.column_stack(
                    (x_coords_graph, y_coords_graph, z_coords_graph))
            )

        # Decode the adjacency matrix edge logits for each graph in the
        # data batch and for each multiedge order network -- in that
        # conventional order for looping
        adj_edges_logits = torch.tensor([], dtype=torch.float).to(batch.device)
        for batch_indx in range(batch_size):
            for multiedge in range(self.multiedge_max):
                if multiedge == 0:
                    z_adj = (
                        z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :self.d_adj_combnd_0]
                    )
                else:
                    z_adj = (
                        z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), getattr(self, f"d_{(multiedge-1):d}"):getattr(self, f"d_{multiedge:d}")]
                    )
                adj_edges_logits = (
                    torch.cat(
                        (adj_edges_logits, triu_adj_edges(decoded_z_to_adj_or_adj_logits(z_adj, False, False, True, False, 0.5))))
                )

        # Decode the edge types and edge contour lengths using the
        # decoded nodal coordinates (which can be considered as learned
        # nodal embeddings)
        edge_type_logits = self.edge_type_decoder(coords, data.edge_index)
        edge_l_cntr = self.edge_l_cntr_decoder(coords, data.edge_index)

        if not self.synthesis_protocol_params and not self.property_descriptors:
            return L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr
        elif self.synthesis_protocol_params and not self.property_descriptors:
            # Extract the region of the latent space attributed to each
            # graph in the data batch, and decode the synthesis protocol
            # parameters
            xi = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            chi = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            en_mean = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            for batch_indx in range(batch_size):
                z_graph = z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :]
                xi[batch_indx] = self.xi_decoder(z_graph)
                chi[batch_indx] = self.chi_decoder(z_graph)
                en_mean[batch_indx] = self.en_mean_decoder(z_graph)
            return (
                L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi,
                chi, en_mean
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            # Extract the region of the latent space attributed to each
            # graph in the data batch, and decode the property 
            # descriptors
            eeel_dobrynin_kappa = (
                torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            )
            eeel_glbl_mean_gamma = (
                torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            )
            for batch_indx in range(batch_size):
                z_graph = z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :]
                eeel_dobrynin_kappa[batch_indx] = (
                    self.eeel_dobrynin_kappa_decoder(z_graph)
                )
                eeel_glbl_mean_gamma[batch_indx] = (
                    self.eeel_glbl_mean_gamma_decoder(z_graph)
                )
            return (
                L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr,
                eeel_dobrynin_kappa, eeel_glbl_mean_gamma
            )
        else:
            # Extract the region of the latent space attributed to each
            # graph in the data batch, and decode the synthesis protocol
            # parameters and the property descriptors
            xi = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            chi = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            en_mean = torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            eeel_dobrynin_kappa = (
                torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            )
            eeel_glbl_mean_gamma = (
                torch.zeros(batch_size, dtype=torch.float).to(batch.device)
            )
            for batch_indx in range(batch_size):
                z_graph = z[(self.n_graph*batch_indx):(self.n_graph*(batch_indx+1)), :]
                xi[batch_indx] = self.xi_decoder(z_graph)
                chi[batch_indx] = self.chi_decoder(z_graph)
                en_mean[batch_indx] = self.en_mean_decoder(z_graph)
                eeel_dobrynin_kappa[batch_indx] = (
                    self.eeel_dobrynin_kappa_decoder(z_graph)
                )
                eeel_glbl_mean_gamma[batch_indx] = (
                    self.eeel_glbl_mean_gamma_decoder(z_graph)
                )
            return (
                L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi,
                chi, en_mean, eeel_dobrynin_kappa, eeel_glbl_mean_gamma
            )
    
    def forward_all(self, z: torch.Tensor):
        # Extract the region of the latent space attributed to the nodal
        # coordinates, and decode the simulation box side length and
        # nodal coordinates
        z_coords_graph = z[:, self.d_adj_0:self.d_0]
        L = self.L_decoder(z_coords_graph)
        flat_coords_graph = self.flat_coords_decoder(z_coords_graph.flatten())
        x_coords_graph = flat_coords_graph[:self.n_graph]
        y_coords_graph = flat_coords_graph[self.n_graph:(2*self.n_graph)]
        z_coords_graph = flat_coords_graph[(2*self.n_graph):]
        coords = torch.column_stack(
            (x_coords_graph, y_coords_graph, z_coords_graph))
        
        # Decode the adjacency matrix and edge index for each multiedge
        # order network and for all multiedge order networks
        adj_multiedges = torch.zeros(
            (self.multiedge_max, self.n_graph, self.n_graph), dtype=torch.long)
        for multiedge in range(self.multiedge_max):
            if multiedge == 0: z_adj = z[:, :self.d_adj_combnd_0]
            else:
                z_adj = (
                    z[:, getattr(self, f"d_{(multiedge-1):d}"):getattr(self, f"d_{multiedge:d}")]
                )
            adj_multiedge = decoded_z_to_adj_or_adj_logits(
                z_adj, True, True, True, False, 0.5)
            adj_multiedges[multiedge, :, :] = adj_multiedge.long()
        adj = torch.sum(adj_multiedges, dim=0)

        edge_index_0 = torch.tensor([], dtype=torch.long)
        edge_index_1 = torch.tensor([], dtype=torch.long)
        edge_l_cntr_edge_index_0 = torch.tensor([], dtype=torch.long)
        edge_l_cntr_edge_index_1 = torch.tensor([], dtype=torch.long)
        adj_multiedges_counter = torch.zeros(
            (self.n_graph, self.n_graph), dtype=torch.long)
        adj_multiedges_triu_indcs = torch.triu_indices(
            self.n_graph, self.n_graph)
        for multiedge in range(self.multiedge_max):
            for triu_indx in range(adj_multiedges_triu_indcs.size(1)):
                node_0 = adj_multiedges_triu_indcs[0, triu_indx].item()
                node_1 = adj_multiedges_triu_indcs[1, triu_indx].item()
                if adj_multiedges[multiedge, node_0, node_1] == 1:
                    adj_multiedges_counter[node_0, node_1] += 1
                    edge_index_0 = torch.cat(
                        (edge_index_0, torch.tensor([node_0], dtype=torch.long)))
                    edge_index_1 = torch.cat(
                        (edge_index_1, torch.tensor([node_1], dtype=torch.long)))
                    if adj[node_0, node_1] > 1 and adj_multiedges_counter[node_0, node_1] > 1:
                        node_0, node_1 = torch.randint(self.n_graph, (2,))
                        node_0 = node_0.item()
                        node_1 = node_1.item()
                    edge_l_cntr_edge_index_0 = torch.cat(
                        (edge_l_cntr_edge_index_0, torch.tensor([node_0], dtype=torch.long)))
                    edge_l_cntr_edge_index_1 = torch.cat(
                        (edge_l_cntr_edge_index_1, torch.tensor([node_1], dtype=torch.long)))
        edge_index = torch.vstack((edge_index_0, edge_index_1))
        edge_l_cntr_edge_index = torch.vstack(
            (edge_l_cntr_edge_index_0, edge_l_cntr_edge_index_1))
        
        # Decode the edge types using the decoded nodal coordinates and
        # edge index
        edge_type_logits = self.edge_type_decoder(coords, edge_index)
        edge_type = torch.bernoulli(torch.sigmoid(edge_type_logits)) # .long() -> apply this after

        # Decode the edge contour lengths using the decoded nodal
        # coordinates and appropriate edge index
        edge_l_cntr = self.edge_l_cntr_decoder(coords, edge_l_cntr_edge_index)

        # Account for repeated interleaven and every-other flipped entry
        # structure of edge_index
        edge_index = torch.repeat_interleave(edge_index, 2, dim=1)
        edge_index[:, 1::2] = torch.flipud(edge_index[:, 1::2])
        edge_type = torch.repeat_interleave(edge_type, 2)
        edge_l_cntr = torch.repeat_interleave(edge_l_cntr, 2)

        if not self.synthesis_protocol_params and not self.property_descriptors:
            return L, coords, edge_index, edge_type, edge_l_cntr
        elif self.synthesis_protocol_params and not self.property_descriptors:
            # Decode the synthesis protocol parameters
            xi = self.xi_decoder(z)
            chi = self.chi_decoder(z)
            en_mean = self.en_mean_decoder(z)
            return (
                xi, chi, en_mean, L, coords, edge_index, edge_type, edge_l_cntr
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            # Decode the property descriptors
            eeel_dobrynin_kappa = self.eeel_dobrynin_kappa_decoder(z)
            eeel_glbl_mean_gamma = self.eeel_glbl_mean_gamma_decoder(z)
            return (
                L, coords, edge_index, edge_type, edge_l_cntr,
                eeel_dobrynin_kappa, eeel_glbl_mean_gamma
            )
        else:
            # Decode the synthesis protocol parameters and the property
            # descriptors
            xi = self.xi_decoder(z)
            chi = self.chi_decoder(z)
            en_mean = self.en_mean_decoder(z)
            eeel_dobrynin_kappa = self.eeel_dobrynin_kappa_decoder(z)
            eeel_glbl_mean_gamma = self.eeel_glbl_mean_gamma_decoder(z)
            return (
                xi, chi, en_mean, L, coords, edge_index, edge_type, edge_l_cntr,
                eeel_dobrynin_kappa, eeel_glbl_mean_gamma
            )
        