import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.apelp_networks_dataset import valid_params
from torch_geometric.nn.models.autoencoder import VGAE
from src.models.apelp_networks_avgae_encoder import apelpAVGAEEncoder
from src.models.apelp_networks_avgae_decoder import apelpAVGAEDecoder
from src.helpers.model_utils import (
    batch_analysis,
    triu_adj_edges,
    apply_threshold,
    eval_func
)

class apelpAVGAE(VGAE):
    """
    Fill in later. Add typehinting in the function calls

    """
    def __init__(self, cfg, n_graph):
        super().__init__(
            apelpAVGAEEncoder(cfg, n_graph), apelpAVGAEDecoder(cfg, n_graph))
        dim, _, _, _, _, _, self.multiedge_max = valid_params(cfg)
        assert dim == 3
        self.n_graph = n_graph
        self.synthesis_protocol_params = cfg.model.synthesis_protocol_params
        self.property_descriptors = cfg.model.property_descriptors
        if cfg.model.regress_loss_func == "mse":
            self.regress_loss_criteron = nn.MSELoss()
        elif cfg.model.regress_loss_func == "l1":
            self.regress_loss_criteron = nn.L1Loss()
        self.edge_loss_criterion = nn.BCEWithLogitsLoss()
        self.binary_class_loss_criterion = nn.BCEWithLogitsLoss()
        self.edges_eval_func = cfg.model.edges_eval_func
        self.edge_type_eval_func = cfg.model.edge_type_eval_func
        self.descriptor_regress_eval_func = (
            cfg.model.descriptor_regress_eval_func
        )
        self.ntwrk_compnts_regress_eval_func = (
            cfg.model.ntwrk_compnts_regress_eval_func
        )
    
    def data_adj_edges(self, data: torch.Tensor):
        # Extract the batch, the number of nodes from the as-provided
        # network for the data batch, the batch size, and the number of 
        # negative edges to sample for each individual pre-batched graph
        # in the data batch
        batch, n, batch_size = batch_analysis(data)
        assert self.n_graph == (n//batch_size)

        adj_diag_indx = torch.arange(self.n_graph).to(batch.device)
        data_adj = (
            torch.zeros((batch_size, self.multiedge_max, self.n_graph, self.n_graph), dtype=torch.float).to(batch.device)
        )
        for multiedge in range(self.multiedge_max):
            edge_index = getattr(data, f"edge_index_{multiedge:d}")
            if edge_index.numel() > 0:
                for edge in range(edge_index.size(1)):
                    node_0 = edge_index[0, edge].item()
                    node_1 = edge_index[1, edge].item()
                    assert (node_0//self.n_graph) == (node_1//self.n_graph)
                    batch_indx = node_0 // self.n_graph
                    node_0 = node_0 % self.n_graph
                    node_1 = node_1 % self.n_graph
                    data_adj[batch_indx, multiedge, node_0, node_1] += 1
        data_adj_edges = torch.tensor([], dtype=torch.float).to(batch.device)
        for batch_indx in range(batch_size):
            for multiedge in range(self.multiedge_max):
                adj = data_adj[batch_indx, multiedge]
                if multiedge == 0: # self-loop correction
                    adj[adj_diag_indx, adj_diag_indx] = torch.clamp(
                        adj[adj_diag_indx, adj_diag_indx], min=0, max=1)
                adj = torch.maximum(adj, adj.t())
                data_adj_edges = torch.cat((data_adj_edges, triu_adj_edges(adj)))

        return data_adj_edges
    
    def recon_loss(self, z: torch.Tensor, data: torch.Tensor):
        # Gather data_adj_edges
        data_adj_edges = self.data_adj_edges(data)

        # Decode batched data graph
        if not self.synthesis_protocol_params and not self.property_descriptors:
            L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr = (
                self.decoder(z, data)
            )
        elif self.synthesis_protocol_params and not self.property_descriptors:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi, chi,
             en_mean) = self.decoder(z, data)
        elif not self.synthesis_protocol_params and self.property_descriptors:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr,
             eeel_dobrynin_kappa, eeel_glbl_mean_gamma) = self.decoder(z, data)
        else:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi, chi,
             en_mean, eeel_dobrynin_kappa, eeel_glbl_mean_gamma) = self.decoder(
                z, data)
        
        # Evaluate regression loss for simulation box side length and
        # nodal coordinates
        L_loss = self.regress_loss_criteron(L, data.L)
        coords_loss = self.regress_loss_criteron(coords, data.coords)

        # Evaluate balanced (via mean reduction) binary cross entropy
        # edge loss (with logits)
        edge_loss = self.edge_loss_criterion(adj_edges_logits, data_adj_edges)

        # Evaluate cross entropy loss for edge types and regression loss
        # for edge contour lengths
        edge_type_loss = self.binary_class_loss_criterion(
            edge_type_logits, data.edge_type)
        edge_l_cntr_loss = self.regress_loss_criteron(
            edge_l_cntr, data.edge_l_cntr)

        # Evaluate reconstruction loss
        recon_loss = (
            L_loss + coords_loss + edge_loss + edge_type_loss + edge_l_cntr_loss
        )
        
        if not self.synthesis_protocol_params and not self.property_descriptors:
            # Return reconstruction loss followed by each of its summed
            # components 
            return (
                recon_loss, L_loss.item(), coords_loss.item(), edge_loss.item(),
                edge_type_loss.item(), edge_l_cntr_loss.item()
            )
        elif self.synthesis_protocol_params and not self.property_descriptors:
            # Evaluate regression loss for the synthesis protocol
            # parameters
            xi_loss = self.regress_loss_criteron(xi, data.xi)
            chi_loss = self.regress_loss_criteron(chi, data.chi)
            en_mean_loss = self.regress_loss_criteron(en_mean, data.en_mean)
            # Evaluate and return reconstruction loss followed by each
            # of its summed components
            recon_loss += xi_loss + chi_loss + en_mean_loss
            return (
                recon_loss, L_loss.item(), coords_loss.item(), edge_loss.item(),
                edge_type_loss.item(), edge_l_cntr_loss.item(), xi_loss.item(),
                chi_loss.item(), en_mean_loss.item()
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            # Evaluate regression loss for the property descriptors
            eeel_dobrynin_kappa_loss = self.regress_loss_criteron(
                eeel_dobrynin_kappa, data.eeel_dobrynin_kappa)
            eeel_glbl_mean_gamma_loss = self.regress_loss_criteron(
                eeel_glbl_mean_gamma, data.eeel_glbl_mean_gamma)
            # Evaluate and return reconstruction loss followed by each
            # of its summed components
            recon_loss += (
                eeel_dobrynin_kappa_loss + eeel_glbl_mean_gamma_loss
            )
            return (
                recon_loss, L_loss.item(), coords_loss.item(), edge_loss.item(),
                edge_type_loss.item(), edge_l_cntr_loss.item(),
                eeel_dobrynin_kappa_loss.item(),
                eeel_glbl_mean_gamma_loss.item()
            )
        else:
            # Evaluate regression loss for the synthesis protocol
            # parameters and the property descriptors
            xi_loss = self.regress_loss_criteron(xi, data.xi)
            chi_loss = self.regress_loss_criteron(chi, data.chi)
            en_mean_loss = self.regress_loss_criteron(en_mean, data.en_mean)
            eeel_dobrynin_kappa_loss = self.regress_loss_criteron(
                eeel_dobrynin_kappa, data.eeel_dobrynin_kappa)
            eeel_glbl_mean_gamma_loss = self.regress_loss_criteron(
                eeel_glbl_mean_gamma, data.eeel_glbl_mean_gamma)
            # Evaluate and return reconstruction loss followed by each
            # of its summed components
            recon_loss += (
                xi_loss + chi_loss + en_mean_loss
                + eeel_dobrynin_kappa_loss + eeel_glbl_mean_gamma_loss
            )
            return (
                recon_loss, L_loss.item(), coords_loss.item(), edge_loss.item(),
                edge_type_loss.item(), edge_l_cntr_loss.item(), xi_loss.item(),
                chi_loss.item(), en_mean_loss.item(),
                eeel_dobrynin_kappa_loss.item(),
                eeel_glbl_mean_gamma_loss.item()
            )
    
    def total_loss(self, beta: float, z: torch.Tensor, data: torch.Tensor):
        # Evaluate beta-KL divergence loss
        beta_kl_loss = beta * self.kl_loss()

        # Evaluate reconstruction loss
        if not self.synthesis_protocol_params and not self.property_descriptors:
            (recon_loss, L_loss, coords_loss, edge_loss, edge_type_loss,
             edge_l_cntr_loss) = self.recon_loss(z, data)
        elif self.synthesis_protocol_params and not self.property_descriptors:
            (recon_loss, L_loss, coords_loss, edge_loss, edge_type_loss,
             edge_l_cntr_loss, xi_loss, chi_loss, en_mean_loss) = (
                self.recon_loss(z, data)
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            (recon_loss, L_loss, coords_loss, edge_loss, edge_type_loss,
             edge_l_cntr_loss, eeel_dobrynin_kappa_loss,
             eeel_glbl_mean_gamma_loss) = self.recon_loss(z, data)
        else:
            (recon_loss, L_loss, coords_loss, edge_loss, edge_type_loss,
             edge_l_cntr_loss, xi_loss, chi_loss, en_mean_loss,
             eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss) = (
                self.recon_loss(z, data)
            )
        
        # Add KL-divergence loss to reconstruction loss to yield total
        # loss
        total_loss = beta_kl_loss + recon_loss

        # Return total loss followed by each of its summed components
        if not self.synthesis_protocol_params and not self.property_descriptors:
            return (
                total_loss, beta_kl_loss.item(), recon_loss.item(), L_loss,
                coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss
            )
        elif self.synthesis_protocol_params and not self.property_descriptors:
            return (
                total_loss, beta_kl_loss.item(), recon_loss.item(), L_loss,
                coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                xi_loss, chi_loss, en_mean_loss
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            return (
                total_loss, beta_kl_loss.item(), recon_loss.item(), L_loss,
                coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
            )
        else:
            return (
                total_loss, beta_kl_loss.item(), recon_loss.item(), L_loss,
                coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                xi_loss, chi_loss, en_mean_loss, eeel_dobrynin_kappa_loss,
                eeel_glbl_mean_gamma_loss
            )
    
    def test(self, z: torch.Tensor, data: torch.Tensor):
        # Gather data_adj_edges
        data_adj_edges = self.data_adj_edges(data)
        
        # Decode batched data graph
        if not self.synthesis_protocol_params and not self.property_descriptors:
            L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr = (
                self.decoder(z, data)
            )
        elif self.synthesis_protocol_params and not self.property_descriptors:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi, chi,
             en_mean) = self.decoder(z, data)
        elif not self.synthesis_protocol_params and self.property_descriptors:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr,
             eeel_dobrynin_kappa, eeel_glbl_mean_gamma) = self.decoder(z, data)
        else:
            (L, coords, adj_edges_logits, edge_type_logits, edge_l_cntr, xi, chi,
             en_mean, eeel_dobrynin_kappa, eeel_glbl_mean_gamma) = self.decoder(
                z, data)
        
        # Evaluate validation/test metrics
        L_true = data.L.detach().cpu().numpy()
        L_pred = L.detach().cpu().numpy()
        L_eval_metric = eval_func(
            L_true, L_pred, self.descriptor_regress_eval_func)
        
        coords_true = data.coords.detach().cpu().numpy().flatten()
        coords_pred = coords.detach().cpu().numpy().flatten()
        coords_eval_metric = eval_func(
            coords_true, coords_pred, self.ntwrk_compnts_regress_eval_func)
        
        edges_true = data_adj_edges.detach().cpu().numpy()
        edges_pred = F.sigmoid(adj_edges_logits).detach().cpu().numpy()
        edges_eval_metric = eval_func(
            edges_true, edges_pred, self.edges_eval_func)
        
        edges_eval_ap_metric = eval_func(edges_true, edges_pred, "ap")
        edges_eval_gini_metric = eval_func(edges_true, edges_pred, "gini")

        edge_type_true = data.edge_type.detach().cpu().numpy()
        edge_type_pred = F.sigmoid(edge_type_logits).detach().cpu().numpy()
        edge_type_eval_metric = eval_func(
            edge_type_true, edge_type_pred, self.edge_type_eval_func)

        edge_type_eval_ap_metric = eval_func(
            edge_type_true, edge_type_pred, "ap")
        edge_type_eval_gini_metric = eval_func(
            edge_type_true, edge_type_pred, "gini")

        edge_l_cntr_true = data.edge_l_cntr.detach().cpu().numpy()
        edge_l_cntr_pred = edge_l_cntr.detach().cpu().numpy()
        edge_l_cntr_eval_metric = eval_func(
            edge_l_cntr_true, edge_l_cntr_pred,
            self.ntwrk_compnts_regress_eval_func)

        total_eval_metric = (
            L_eval_metric + coords_eval_metric + edges_eval_metric
            + edge_type_eval_metric + edge_l_cntr_eval_metric
        )
        
        if not self.synthesis_protocol_params and not self.property_descriptors:
            return (
                total_eval_metric, L_eval_metric, coords_eval_metric,
                edges_eval_metric, edge_type_eval_metric,
                edge_l_cntr_eval_metric, edges_eval_ap_metric,
                edges_eval_gini_metric, edge_type_eval_ap_metric,
                edge_type_eval_gini_metric
            )
        elif self.synthesis_protocol_params and not self.property_descriptors:
            xi_true = data.xi.detach().cpu().numpy()
            xi_pred = xi.detach().cpu().numpy()
            xi_eval_metric = eval_func(
                xi_true, xi_pred, self.descriptor_regress_eval_func)
            
            chi_true = data.chi.detach().cpu().numpy()
            chi_pred = chi.detach().cpu().numpy()
            chi_eval_metric = eval_func(
                chi_true, chi_pred, self.descriptor_regress_eval_func)
            
            en_mean_true = data.en_mean.detach().cpu().numpy()
            en_mean_pred = en_mean.detach().cpu().numpy()
            en_mean_eval_metric = eval_func(
                en_mean_true, en_mean_pred, self.descriptor_regress_eval_func)
            
            total_eval_metric += (
                xi_eval_metric + chi_eval_metric + en_mean_eval_metric
            )

            return (
                total_eval_metric, L_eval_metric, coords_eval_metric,
                edges_eval_metric, edge_type_eval_metric,
                edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                en_mean_eval_metric, edges_eval_ap_metric,
                edges_eval_gini_metric, edge_type_eval_ap_metric,
                edge_type_eval_gini_metric
            )
        elif not self.synthesis_protocol_params and self.property_descriptors:
            eeel_dobrynin_kappa_true = (
                data.eeel_dobrynin_kappa.detach().cpu().numpy()
            )
            eeel_dobrynin_kappa_pred = (
                eeel_dobrynin_kappa.detach().cpu().numpy()
            )
            eeel_dobrynin_kappa_eval_metric = eval_func(
                eeel_dobrynin_kappa_true, eeel_dobrynin_kappa_pred,
                self.descriptor_regress_eval_func)
            
            eeel_glbl_mean_gamma_true = (
                data.eeel_glbl_mean_gamma.detach().cpu().numpy()
            )
            eeel_glbl_mean_gamma_pred = (
                eeel_glbl_mean_gamma.detach().cpu().numpy()
            )
            eeel_glbl_mean_gamma_eval_metric = eval_func(
                eeel_glbl_mean_gamma_true, eeel_glbl_mean_gamma_pred,
                self.descriptor_regress_eval_func)
            
            total_eval_metric += (
                eeel_dobrynin_kappa_eval_metric
                + eeel_glbl_mean_gamma_eval_metric
            )

            return (
                total_eval_metric, L_eval_metric, coords_eval_metric,
                edges_eval_metric, edge_type_eval_metric,
                edge_l_cntr_eval_metric, eeel_dobrynin_kappa_eval_metric,
                eeel_glbl_mean_gamma_eval_metric, edges_eval_ap_metric,
                edges_eval_gini_metric, edge_type_eval_ap_metric,
                edge_type_eval_gini_metric
            )
        else:
            xi_true = data.xi.detach().cpu().numpy()
            xi_pred = xi.detach().cpu().numpy()
            xi_eval_metric = eval_func(
                xi_true, xi_pred, self.descriptor_regress_eval_func)
            
            chi_true = data.chi.detach().cpu().numpy()
            chi_pred = chi.detach().cpu().numpy()
            chi_eval_metric = eval_func(
                chi_true, chi_pred, self.descriptor_regress_eval_func)
            
            en_mean_true = data.en_mean.detach().cpu().numpy()
            en_mean_pred = en_mean.detach().cpu().numpy()
            en_mean_eval_metric = eval_func(
                en_mean_true, en_mean_pred, self.descriptor_regress_eval_func)
            
            eeel_dobrynin_kappa_true = (
                data.eeel_dobrynin_kappa.detach().cpu().numpy()
            )
            eeel_dobrynin_kappa_pred = (
                eeel_dobrynin_kappa.detach().cpu().numpy()
            )
            eeel_dobrynin_kappa_eval_metric = eval_func(
                eeel_dobrynin_kappa_true, eeel_dobrynin_kappa_pred,
                self.descriptor_regress_eval_func)
            
            eeel_glbl_mean_gamma_true = (
                data.eeel_glbl_mean_gamma.detach().cpu().numpy()
            )
            eeel_glbl_mean_gamma_pred = (
                eeel_glbl_mean_gamma.detach().cpu().numpy()
            )
            eeel_glbl_mean_gamma_eval_metric = eval_func(
                eeel_glbl_mean_gamma_true, eeel_glbl_mean_gamma_pred,
                self.descriptor_regress_eval_func)
            
            total_eval_metric += (
                xi_eval_metric + chi_eval_metric + en_mean_eval_metric
                + eeel_dobrynin_kappa_eval_metric
                + eeel_glbl_mean_gamma_eval_metric
            )

            return (
                total_eval_metric, L_eval_metric, coords_eval_metric,
                edges_eval_metric, edge_type_eval_metric,
                edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                en_mean_eval_metric, eeel_dobrynin_kappa_eval_metric,
                eeel_glbl_mean_gamma_eval_metric, edges_eval_ap_metric,
                edges_eval_gini_metric, edge_type_eval_ap_metric,
                edge_type_eval_gini_metric
            )