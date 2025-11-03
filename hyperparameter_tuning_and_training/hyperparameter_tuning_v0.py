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
    load_early_stopping
)
from src.file_io.file_io import (
    chkpnt_filepath_str,
    early_stop_filepath_str,
    chkpnt_filename_str,
    early_stop_filename_str
)
from src.helpers.model_utils import cyclic_beta_kl_div_annealing

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Boilerplate setup
    torch.manual_seed(cfg.general.seed)
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # Load in the model, optimizer, and starting epoch
    model, optimizer, start_epoch = load_model(cfg, device)

    # Load in the best model state dictionary and other parameters
    # associated with early stopping
    (best_model_state_dict, min_total_eval_metric, min_total_eval_metric_epoch,
     patience_counter, start_epoch_early_stop) = load_early_stopping(cfg)
    
    if start_epoch != start_epoch_early_stop:
        error_str = (
            "The checkpoint start_epoch value is not equal to the "
            + "analogous early stopping start_epoch_early_stop value. "
            + "This implies that the simultaneous saving of checkpoint "
            + "and early stopping data was somehow interrupted. "
            + "Consider restarting this hyperparameter tuning run from "
            + "scratch."
        )
        raise ValueError(error_str)

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
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 9))
        elif synthesis_protocol_params and not property_descriptors:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 12))
        elif not synthesis_protocol_params and property_descriptors:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 11))
        else:
            total_loss_compnts = np.zeros((cfg.train.n_epochs, 14))
        kl_loss = np.zeros(cfg.train.n_epochs)
    else:
        total_loss_compnts = np.loadtxt(total_loss_compnts_filename)
        kl_loss = np.loadtxt(kl_loss_filename)
        total_loss_compnts[start_epoch] *= 0
        kl_loss[start_epoch] *= 0
    # Initialize trackers for training loss for logging
    if not synthesis_protocol_params and not property_descriptors:
        total_loss_compnts_tracker = np.zeros(9)
    elif synthesis_protocol_params and not property_descriptors:
        total_loss_compnts_tracker = np.zeros(12)
    elif not synthesis_protocol_params and property_descriptors:
        total_loss_compnts_tracker = np.zeros(11)
    else: total_loss_compnts_tracker = np.zeros(14)
    kl_loss_tracker = 0
    batch_count = 0
    
    # Gather parameters associated with validation metrics
    total_eval_compnts_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-total_eval_compnts.dat"
    )
    edges_eval_gini_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-edges_eval_gini.dat"
    )
    kl_div_filename = (
        chkpnt_filename_str(aelp_network, cfg.label) + "-kl_div.dat"
    )
    if start_epoch == 0:
        if not synthesis_protocol_params and not property_descriptors:
            total_eval_compnts = np.zeros((cfg.train.n_epochs, 6))
        elif synthesis_protocol_params and not property_descriptors:
            total_eval_compnts = np.zeros((cfg.train.n_epochs, 9))
        elif not synthesis_protocol_params and property_descriptors:
            total_eval_compnts = np.zeros((cfg.train.n_epochs, 8))
        else:
            total_eval_compnts = np.zeros((cfg.train.n_epochs, 11))
        edges_eval_gini = np.zeros(cfg.train.n_epochs)
        kl_div = np.zeros(cfg.train.n_epochs)
    else:
        total_eval_compnts = np.loadtxt(total_eval_compnts_filename)
        edges_eval_gini = np.loadtxt(edges_eval_gini_filename)
        kl_div = np.loadtxt(kl_div_filename)
        total_eval_compnts[start_epoch] *= 0
        edges_eval_gini[start_epoch] *= 0
        kl_div[start_epoch] *= 0
    
    # Initialize early stopping boolean flag
    stop = False

    # Number of batches in the training and validation dataloaders
    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)

    # Training and validation loop for hyperparameter tuning
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
        
        # Set model to training mode
        model.train()
        
        # Evaluate each batched data graph in the training dataloader
        for indx, data in enumerate(train_dataloader):
            # Extract batched data to the GPU
            data = data.to(device)
            
            # Encode batched data to a sampled latent space
            z = model.encode(data)
            
            # Calculate total training loss components by decoding the
            # sampled latent space 
            if not synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 pos_edge_loss, neg_edge_loss, edge_type_loss, edge_l_cntr_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, pos_edge_loss, neg_edge_loss,
                        edge_type_loss, edge_l_cntr_loss
                    ])
            elif synthesis_protocol_params and not property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 pos_edge_loss, neg_edge_loss, edge_type_loss, edge_l_cntr_loss,
                 xi_loss, chi_loss, en_mean_loss) = model.total_loss(
                    beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, pos_edge_loss, neg_edge_loss,
                        edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                        en_mean_loss
                    ])
            elif not synthesis_protocol_params and property_descriptors:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 pos_edge_loss, neg_edge_loss, edge_type_loss, edge_l_cntr_loss,
                 eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss) = (
                    model.total_loss(beta[epoch], z, data)
                )
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, pos_edge_loss, neg_edge_loss,
                        edge_type_loss, edge_l_cntr_loss,
                        eeel_dobrynin_kappa_loss, eeel_glbl_mean_gamma_loss
                    ])
            else:
                (total_loss, beta_kl_loss, recon_loss, L_loss, coords_loss,
                 pos_edge_loss, neg_edge_loss, edge_type_loss, edge_l_cntr_loss,
                 xi_loss, chi_loss, en_mean_loss, eeel_dobrynin_kappa_loss,
                 eeel_glbl_mean_gamma_loss) = model.total_loss(
                    beta[epoch], z, data)
                total_loss_compnts_batch = np.asarray(
                    [
                        total_loss.item(), beta_kl_loss, recon_loss, L_loss,
                        coords_loss, pos_edge_loss, neg_edge_loss,
                        edge_type_loss, edge_l_cntr_loss, xi_loss, chi_loss,
                        en_mean_loss, eeel_dobrynin_kappa_loss,
                        eeel_glbl_mean_gamma_loss
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
                        + f"pos_edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"neg_edge_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[8]:.3f}, "
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
                        + f"pos_edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"neg_edge_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[11]:.3f}, "
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
                        + f"pos_edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"neg_edge_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[10]:.3f}, "
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
                        + f"pos_edge_loss: {total_loss_compnts_tracker[5]:.3f}, "
                        + f"neg_edge_loss: {total_loss_compnts_tracker[6]:.3f}, "
                        + f"edge_type_loss: {total_loss_compnts_tracker[7]:.3f}, "
                        + f"edge_l_cntr_loss: {total_loss_compnts_tracker[8]:.3f}, "
                        + f"xi_loss: {total_loss_compnts_tracker[9]:.3f}, "
                        + f"chi_loss: {total_loss_compnts_tracker[10]:.3f}, "
                        + f"en_mean_loss: {total_loss_compnts_tracker[11]:.3f}, "
                        + f"eeel_dobrynin_kappa_loss: {total_loss_compnts_tracker[12]:.3f}, "
                        + f"eeel_glbl_mean_gamma_loss: {total_loss_compnts_tracker[13]:.3f}, "
                        + f"kl_loss: {kl_loss_tracker:.3f}\n"
                    )
                log.write(log_str)
                log.flush()
                total_loss_compnts_tracker *= 0
                kl_loss_tracker *= 0
                batch_count = 0
        log.close()

        # Normalize total training loss by number of training data
        # batches
        total_loss_compnts[epoch] /= n_train_batches
        kl_loss[epoch] /= n_train_batches

        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            # Evaluate each batched data graph in the validation
            # dataloader
            for indx, data in enumerate(val_dataloader):
                # Extract batched data to the GPU
                data = data.to(device)
                
                # Encode batched data to a sampled latent space
                z = model.encode(data)

                # Calculate total validation metric components by
                # decoding the sampled latent space
                if not synthesis_protocol_params and not property_descriptors:
                    (total_eval_metric, L_eval_metric, coords_eval_metric,
                     edges_eval_average_precision_metric, edge_type_eval_metric,
                     edge_l_cntr_eval_metric, edges_eval_gini_metric) = (
                        model.test(z, data)
                    )
                    total_eval_compnts_batch = np.asarray(
                        [
                            total_eval_metric, L_eval_metric, coords_eval_metric,
                            edges_eval_average_precision_metric,
                            edge_type_eval_metric, edge_l_cntr_eval_metric
                        ])
                elif synthesis_protocol_params and not property_descriptors:
                    (total_eval_metric, L_eval_metric, coords_eval_metric,
                     edges_eval_average_precision_metric, edge_type_eval_metric,
                     edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                     en_mean_eval_metric, edges_eval_gini_metric) = model.test(
                        z, data)
                    total_eval_compnts_batch = np.asarray(
                        [
                            total_eval_metric, L_eval_metric, coords_eval_metric,
                            edges_eval_average_precision_metric,
                            edge_type_eval_metric, edge_l_cntr_eval_metric,
                            xi_eval_metric, chi_eval_metric, en_mean_eval_metric
                        ])
                elif not synthesis_protocol_params and property_descriptors:
                    (total_eval_metric, L_eval_metric, coords_eval_metric,
                     edges_eval_average_precision_metric, edge_type_eval_metric,
                     edge_l_cntr_eval_metric, eeel_dobrynin_kappa_eval_metric,
                     eeel_glbl_mean_gamma_eval_metric, edges_eval_gini_metric) = (
                        model.test(z, data)
                    )
                    total_eval_compnts_batch = np.asarray(
                        [
                            total_eval_metric, L_eval_metric, coords_eval_metric,
                            edges_eval_average_precision_metric,
                            edge_type_eval_metric, edge_l_cntr_eval_metric,
                            eeel_dobrynin_kappa_eval_metric,
                            eeel_glbl_mean_gamma_eval_metric
                        ])
                else:
                    (total_eval_metric, L_eval_metric, coords_eval_metric,
                     edges_eval_average_precision_metric, edge_type_eval_metric,
                     edge_l_cntr_eval_metric, xi_eval_metric, chi_eval_metric,
                     en_mean_eval_metric, eeel_dobrynin_kappa_eval_metric,
                     eeel_glbl_mean_gamma_eval_metric, edges_eval_gini_metric) = (
                        model.test(z, data)
                    )
                    total_eval_compnts_batch = np.asarray(
                        [
                            total_eval_metric, L_eval_metric, coords_eval_metric,
                            edges_eval_average_precision_metric,
                            edge_type_eval_metric, edge_l_cntr_eval_metric,
                            xi_eval_metric, chi_eval_metric, en_mean_eval_metric,
                            eeel_dobrynin_kappa_eval_metric,
                            eeel_glbl_mean_gamma_eval_metric
                        ])
                
                # Update total validation metric loss components
                total_eval_compnts[epoch] += total_eval_compnts_batch
                edges_eval_gini[epoch] += edges_eval_gini_metric
                kl_div[epoch] += model.kl_loss().item()
            
        # Normalize total validation metrics by number of validation
        # data batches
        total_eval_compnts[epoch] /= n_val_batches
        edges_eval_gini[epoch] /= n_val_batches
        kl_div[epoch] /= n_val_batches

        # Evaluate early stopping on total validation metric
        total_eval_metric = total_eval_compnts[epoch, 0]
        if total_eval_metric < min_total_eval_metric:
            min_total_eval_metric = total_eval_metric
            min_total_eval_metric_epoch = epoch
            patience_counter = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience: stop = True
        
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
        
        # Save parameters associated with validation metrics
        np.savetxt(total_eval_compnts_filename, total_eval_compnts)
        np.savetxt(edges_eval_gini_filename, edges_eval_gini)
        np.savetxt(kl_div_filename, kl_div)
        # Save early stopping
        early_stop_dict = {
            "best_model_state_dict": best_model_state_dict,
            "min_total_eval_metric": min_total_eval_metric,
            "min_total_eval_metric_epoch": min_total_eval_metric_epoch,
            "patience_counter": patience_counter,
            "epoch": epoch
        }
        torch.save(
            early_stop_dict,
            early_stop_filename_str(aelp_network, cfg.label)+".model")
        
        # If early stopping criteria are satisfied, then break out of
        # training and validation loop for hyperparameter tuning
        if stop:
            early_stop_dict = {
                "best_model_state_dict": best_model_state_dict,
                "min_total_eval_metric": min_total_eval_metric,
                "min_total_eval_metric_epoch": min_total_eval_metric_epoch,
                "patience_counter": patience_counter,
                "early_stop_epoch": epoch,
                "epoch": epoch
            }
            torch.save(
                early_stop_dict,
                early_stop_filename_str(aelp_network, cfg.label)+".model")
            early_stop_str = f"Early stopping at epoch {epoch:d}"
            log = open(chkpnt_filename_str(aelp_network, cfg.label)+".log", 'a')
            log.write(early_stop_str)
            log.flush()
            log.close()
            print(early_stop_str)
            break

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Hyperparameter tuning code took {execution_time} seconds to run")