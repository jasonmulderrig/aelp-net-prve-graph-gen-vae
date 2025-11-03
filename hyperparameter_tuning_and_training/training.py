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
    load_model
)
from src.file_io.file_io import (
    chkpnt_filepath_str,
    chkpnt_filename_str
)
from src.helpers.model_utils import (
    batch_analysis,
    cyclic_beta_kl_div_annealing
)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    torch.manual_seed(cfg.general.seed)
    torch.autograd.set_detect_anomaly(True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    
    # Load in the training, validation, and testing dataloaders
    train_dataloader, val_dataloader, test_dataloader = load_dataloaders(cfg)

    # Determine the number of nodes in each graph in the dataset
    data = next(iter(train_dataloader))
    batch, n, batch_size = batch_analysis(data)
    n_graph = n // batch_size
    assert (n%batch_size) == 0
    
    # Load in the model, optimizer, and starting epoch
    model, optimizer, start_epoch = load_model(cfg, n_graph, device)
    
    # Gather parameters associated with training
    lr_filename = chkpnt_filename_str(aelp_network, cfg.label) + "-lr.dat"
    beta_filename = chkpnt_filename_str(aelp_network, cfg.label) + "-beta.dat"
    if start_epoch == 0:
        lr = np.zeros(cfg.train.n_epochs)
        beta = np.zeros(cfg.train.n_epochs)
    else:
        lr = np.loadtxt(lr_filename)
        beta = np.loadtxt(beta_filename)
        lr[start_epoch] *= 0
        beta[start_epoch] *= 0
    
    # Gather parameters associated with training loss
    total_loss_compnts_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-total_loss_compnts.dat"
    )
    kl_loss_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-kl_loss.dat"
    )
    if start_epoch == 0:
        if not synthesis_protocol_params and not property_descriptors:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 8))
        elif synthesis_protocol_params and not property_descriptors:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 11))
        elif not synthesis_protocol_params and property_descriptors:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 10))
        else:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 13))
        kl_loss = np.zeros(cfg.train.n_epochs)
    else:
        total_loss_compnts = np.loadtxt(total_loss_compnts_filename)
        kl_loss = np.loadtxt(kl_loss_filename)
        total_loss_compnts[start_epoch] *= 0
        kl_loss[start_epoch] *= 0
    # Initialize trackers for training loss for logging
    if not synthesis_protocol_params and not property_descriptors:
        total_loss_compnts_tracker = np.zeros(8)
    elif synthesis_protocol_params and not property_descriptors:
        total_loss_compnts_tracker = np.zeros(11)
    elif not synthesis_protocol_params and property_descriptors:
        total_loss_compnts_tracker = np.zeros(10)
    else: total_loss_compnts_tracker = np.zeros(13)
    kl_loss_tracker = 0
    batch_count = 0
    
    # Number of batches in the training, validation, and testing
    # dataloaders
    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)
    n_test_batches = len(test_dataloader)

    # Set model to training mode
    model.train()

    # Training loop
    for epoch in range(start_epoch, cfg.train.n_epochs):
        # Current learning rate and KL divergence beta
        lr[epoch] = optimizer.param_groups[0]["lr"]
        beta[epoch] = cyclic_beta_kl_div_annealing(
            epoch, cfg.train.n_epochs_per_anneal_cycle,
            cfg.train.rampup_anneal_cycle_ratio, cfg.train.max_beta)
        print_str = (
            f"Starting epoch {epoch:d} with learning rate "
            + f"{lr[epoch]:.6f} and KL div beta {beta[epoch]:.6f}"
        )
        print(print_str)
        log = open(chkpnt_filename_str(aelp_network, cfg.label)+".log", 'a')
        log_str = (
            f"Epoch: {epoch:d}, learning rate: {lr[epoch]:.6f}, beta: "
            + f"{beta[epoch]:.6f}\n"
        )
        log.write(log_str)
        log.flush()
        
        # Evaluate each batched data graph in the training dataloader
        log.write("Training dataset\n")
        log.flush()
        for indx, data in enumerate(train_dataloader):
            # Extract batched data to the GPU
            data = data.to(device)
            
            # Encode batched data to a sampled latent space
            z = model.encode(data)
            
            # Calculate total training loss components by decoding the
            # sampled latent space 
            if not synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss
                    ])
            elif synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                 en_mean_loss) = model.total_loss(beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        xi_loss, chi_loss, en_mean_loss
                    ])
            elif not synthesis_protocol_params and property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss,
                 eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
                    ])
            else:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                 en_mean_loss, eeel_dobrynin_kappa_loss,
                 eeel_glbl_mean_gamma_loss) = model.total_loss(
                    beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        xi_loss, chi_loss, en_mean_loss,
                        eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
                    ])
            
            # Evaluate vanilla KL divergence loss without beta weighting
            kl_loss_batch = model.kl_loss().item()
            
            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm_max)
            optimizer.step()

            # Update and save the tracking of the total training loss
            # components
            total_loss_compnts[epoch] += total_loss_compnts_batch
            total_loss_compnts_tracker += total_loss_compnts_batch
            kl_loss[epoch] += kl_loss_batch
            kl_loss_tracker += kl_loss_batch
            batch_count += 1
            if ((indx+1)%cfg.train.loss_log_batch_indx) == 0 or (indx+1) == n_train_batches:
                total_loss_compnts_tracker /= batch_count
                kl_loss_tracker /= batch_count
                if not synthesis_protocol_params and not property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                elif synthesis_protocol_params and not property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                elif not synthesis_protocol_params and property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                else:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[11]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[12]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                log.write(log_str)
                log.flush()
                total_loss_compnts_tracker *= 0
                kl_loss_tracker *= 0
                batch_count = 0
        
        # Evaluate each batched data graph in the validation dataloader
        log.write("Validation dataset\n")
        log.flush()
        for indx, data in enumerate(val_dataloader):
            # Extract batched data to the GPU
            data = data.to(device)
            
            # Encode batched data to a sampled latent space
            z = model.encode(data)
            
            # Calculate total training loss components by decoding the
            # sampled latent space 
            if not synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss
                    ])
            elif synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                 en_mean_loss) = model.total_loss(beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        xi_loss, chi_loss, en_mean_loss
                    ])
            elif not synthesis_protocol_params and property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss,
                 eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
                    ])
            else:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 edge_loss, edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                 en_mean_loss, eeel_dobrynin_kappa_loss,
                 eeel_glbl_mean_gamma_loss) = model.total_loss(
                    beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, edge_loss, edge_type_loss, edge_l_cntr_loss,
                        xi_loss, chi_loss, en_mean_loss,
                        eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
                    ])
            
            # Evaluate vanilla KL divergence loss without beta weighting
            kl_loss_batch = model.kl_loss().item()
            
            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm_max)
            optimizer.step()

            # Update and save the tracking of the total training loss
            # components
            total_loss_compnts[epoch] += total_loss_compnts_batch
            total_loss_compnts_tracker += total_loss_compnts_batch
            kl_loss[epoch] += kl_loss_batch
            kl_loss_tracker += kl_loss_batch
            batch_count += 1
            if ((indx+1)%cfg.train.loss_log_batch_indx) == 0 or (indx+1) == n_val_batches:
                total_loss_compnts_tracker /= batch_count
                kl_loss_tracker /= batch_count
                if not synthesis_protocol_params and not property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                elif synthesis_protocol_params and not property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                elif not synthesis_protocol_params and property_descriptors:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                else:
                    log_str = (
                        f"Batch {indx:d}, "
                        + f"total_loss: {total_loss_compnts_tracker[0]:.3f}, "
                        + f"beta_kl_loss: {total_loss_compnts_tracker[1]:.3f}, "
                        + f"recon_loss: {total_loss_compnts_tracker[2]:.3f}, "
                        + f"L_loss: {total_loss_compnts_tracker[3]:.3f}, "
                        + f"coords_loss: {total_loss_compnts_tracker[4]:.3f}, "
                        + f"edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[11]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[12]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                log.write(log_str)
                log.flush()
                total_loss_compnts_tracker *= 0
                kl_loss_tracker *= 0
                batch_count = 0
        log.close()

        # Normalize total training loss by number of training and
        # validation data batches
        total_loss_compnts[epoch] /= (n_train_batches+n_val_batches)
        kl_loss[epoch] /= (n_train_batches+n_val_batches)

        # Save parameters associated with training
        np.savetxt(lr_filename, lr)
        np.savetxt(beta_filename, beta)
        # Save parameters associated with training loss
        np.savetxt(total_loss_compnts_filename, total_loss_compnts)
        np.savetxt(kl_loss_filename, kl_loss)
        # Save the checkpoint
        chkpnt_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        torch.save(
            chkpnt_dict, chkpnt_filename_str(aelp_network, cfg.label)+".model")

    # Gather parameters associated with test validation metrics
    total_eval_compnts_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-total_eval_compnts.dat"
    )
    edges_eval_ap_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-edges_eval_ap.dat"
    )
    edges_eval_gini_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-edges_eval_gini.dat"
    )
    edge_type_eval_ap_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-edge_type_eval_ap.dat"
    )
    edge_type_eval_gini_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-edge_type_eval_gini.dat"
    )
    kl_div_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-kl_div.dat"
    )
    if not synthesis_protocol_params and not property_descriptors:
        total_eval_compnts = np.zeros(6)
    elif synthesis_protocol_params and not property_descriptors:
        total_eval_compnts = np.zeros(9)
    elif not synthesis_protocol_params and property_descriptors:
        total_eval_compnts = np.zeros(8)
    else: total_eval_compnts = np.zeros(11)
    edges_eval_ap = 0.0
    edges_eval_gini = 0.0
    edge_type_eval_ap = 0.0
    edge_type_eval_gini = 0.0
    kl_div = 0.0
    
    # Set model to evaluation mode
    print("Evaluating test validation metrics")
    model.eval()
    with torch.no_grad():
        # Evaluate each batched data graph in the test dataloader
        for indx, data in enumerate(test_dataloader):
            # Extract batched data to the GPU
            data = data.to(device)
            
            # Encode batched data to a sampled latent space
            z = model.encode(data)

            # Calculate total validation metric components by decoding
            # the sampled latent space
            if not synthesis_protocol_params and not property_descriptors:
                (total_eval_metric, L_eval_metric, coords_eval_metric,
                    edges_eval_metric, edge_type_eval_metric,
                    edge_l_cntr_eval_metric, edges_eval_ap_metric,
                    edges_eval_gini_metric, edge_type_eval_ap_metric,
                    edge_type_eval_gini_metric) = model.test(z, data)
                total_eval_compnts_batch = np.asarray(
                    [
                        total_eval_metric, L_eval_metric, coords_eval_metric,
                        edges_eval_metric, edge_type_eval_metric,
                        edge_l_cntr_eval_metric
                    ])
            elif synthesis_protocol_params and not property_descriptors:
                (total_eval_metric, L_eval_metric, coords_eval_metric,
                    edges_eval_metric, edge_type_eval_metric,
                    edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                    en_mean_eval_metric, edges_eval_ap_metric,
                    edges_eval_gini_metric, edge_type_eval_ap_metric,
                    edge_type_eval_gini_metric) = model.test(z, data)
                total_eval_compnts_batch = np.asarray(
                    [
                        total_eval_metric, L_eval_metric, coords_eval_metric,
                        edges_eval_metric, edge_type_eval_metric,
                        edge_l_cntr_eval_metric, xi_eval_metric,
                        chi_eval_metric, en_mean_eval_metric
                    ])
            elif not synthesis_protocol_params and property_descriptors:
                (total_eval_metric, L_eval_metric, coords_eval_metric,
                    edges_eval_metric, edge_type_eval_metric,
                    edge_l_cntr_eval_metric, eeel_dobrynin_kappa_eval_metric,
                    eeel_glbl_mean_gamma_eval_metric, edges_eval_ap_metric,
                    edges_eval_gini_metric, edge_type_eval_ap_metric,
                    edge_type_eval_gini_metric) = model.test(z, data)
                total_eval_compnts_batch = np.asarray(
                    [
                        total_eval_metric, L_eval_metric, coords_eval_metric,
                        edges_eval_metric, edge_type_eval_metric,
                        edge_l_cntr_eval_metric,
                        eeel_dobrynin_kappa_eval_metric,
                        eeel_glbl_mean_gamma_eval_metric
                    ])
            else:
                (total_eval_metric, L_eval_metric, coords_eval_metric,
                    edges_eval_metric, edge_type_eval_metric,
                    edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                    en_mean_eval_metric, eeel_dobrynin_kappa_eval_metric,
                    eeel_glbl_mean_gamma_eval_metric, edges_eval_ap_metric,
                    edges_eval_gini_metric, edge_type_eval_ap_metric,
                    edge_type_eval_gini_metric) = model.test(z, data)
                total_eval_compnts_batch = np.asarray(
                    [
                        total_eval_metric, L_eval_metric, coords_eval_metric,
                        edges_eval_metric, edge_type_eval_metric,
                        edge_l_cntr_eval_metric, xi_eval_metric,
                        chi_eval_metric, en_mean_eval_metric,
                        eeel_dobrynin_kappa_eval_metric,
                        eeel_glbl_mean_gamma_eval_metric
                    ])
            
            # Update total validation metric loss components
            total_eval_compnts += total_eval_compnts_batch
            edges_eval_ap += edges_eval_ap_metric
            edges_eval_gini += edges_eval_gini_metric
            edge_type_eval_ap += edge_type_eval_ap_metric
            edge_type_eval_gini += edge_type_eval_gini_metric
            kl_div += model.kl_loss().item()
        
    # Normalize total validation metrics by number of test data batches
    total_eval_compnts /= n_test_batches
    edges_eval_ap /= n_test_batches
    edges_eval_gini /= n_test_batches
    edge_type_eval_ap /= n_test_batches
    edge_type_eval_gini /= n_test_batches
    kl_div /= n_test_batches
    
    # Save parameters associated with validation metrics
    np.savetxt(total_eval_compnts_filename, total_eval_compnts)
    np.savetxt(edges_eval_ap_filename, np.asarray([edges_eval_ap]))
    np.savetxt(edges_eval_gini_filename, np.asarray([edges_eval_gini]))
    np.savetxt(edge_type_eval_ap_filename, np.asarray([edge_type_eval_ap]))
    np.savetxt(edge_type_eval_gini_filename, np.asarray([edge_type_eval_gini]))
    np.savetxt(kl_div_filename, np.asarray([kl_div]))

if __name__ == "__main__":
    # import time
    
    # start_time = time.perf_counter()
    main()
    # end_time = time.perf_counter()

    # execution_time = end_time - start_time
    # print(f"Training code took {execution_time} seconds to run")