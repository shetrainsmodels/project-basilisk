import os, sys
from MambaSSL_JEPA_Model import MambaJEPA, HARMambaConfig
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from test_mamba import test_model 
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from mamba_ssm.modules.mamba2 import Mamba2
import mamba_ssm.modules.mamba2 as mm
from data.preprocessing import fit_labelencoder, Dataset_HAR
from datetime import datetime
import torch.nn.functional as F
import mamba_ssm
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import set_seed
import argparse
from torch.profiler import profile, ProfilerActivity
import time
import math
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import random

print(f"Mamba version: {mamba_ssm.__version__}")
print(f"Path: {mm.__file__}")

parser = argparse.ArgumentParser(description = "J_mamba")
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--fold", type = int, required = True)
args = parser.parse_args()
if args.dataset == "OPP":
    if args.fold in [1, 2, 3, 4]:
        training_files, validation_files, test_files = data_split_OPP(args.fold)
    else:
        raise ValueError(f"Fold must be 1, 2, 3 or 4. Got {args.fold}")
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, verbose = True)

#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model_PRETRAIN_JEPA(model, val_loader, device, criterion):
    '''
    Validation: avg latent prediction loss, with fixed masks each epoch.
    Also returns target-embedding std as a collapse monitor.
    '''    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    emb_stds = []
    mean_coss = []
    
    with torch.random.fork_rng():
        torch.manual_seed(42)
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(device, non_blocking = True)
            target_embeddings, targets, _, _, _, predictions = model(x_batch)
            loss = criterion (predictions, targets)

            bsize = x_batch.size(0)
            total_loss += loss.item() * bsize
            total_samples += bsize

            flat_emb = target_embeddings.reshape(-1, target_embeddings.size(-1)) # [B*18, 384]
            #Monitor 1: per ft std across all tkns (collapse -> 0)
            emb_stds.append(flat_emb.std(dim = 0).mean().item())
            #Monitor 2: avg cosine similarity bt tokens (collapse -> 1)
            normed = torch.nn.functional.normalize(flat_emb, dim = -1)
            mean_coss.append(normed.mean(dim = 0).norm().pow(2).item())
 
    return total_loss / total_samples, sum(emb_stds) / len(emb_stds), sum(mean_coss) / len(mean_coss)

#  ---------------------------------------------------- TRAINING ---------------------------------------------------
for seed in [42, 58, 7, 128, 92]:
    g = set_seed(seed)
    train_loader, val_loader, test_loader, label_encoder = make_loaders_OPP(X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)

    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HARMambaConfig()
    model = MambaJEPA(config, mask_ratio = 1/3, t_l = 3, use_pe = True, drop = True)
    model.to(device, non_blocking = True)
    # ----------------------
    num_epochs = 35
    warmup_Epochs = 5
    lr = 0.0006
    patience = 8
    criterion = nn.SmoothL1Loss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]   # excludes frozen target encoder
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: #skip frozen targer encoder
            continue
        if p.ndim <= 1 or name.endswith("pe") or "predictor_pe" in name or "mask_token" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    #### Scheduler ####
    def lr_lambda(epoch):
        if epoch < warmup_Epochs:
            return (epoch + 1)/warmup_Epochs
        progress = (epoch - warmup_Epochs)/max(1, num_epochs - warmup_Epochs)
        min_ratio = 1e-6 / lr
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}], lr = lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # EMA momentum schedule: 0.996 -> 1.0 linearly over all steps (I-JEPA style)
    total_steps = num_epochs * len(train_loader)
    m_start, m_end = 0.996, 1.0
    global_step = 0

    #  ----------- TRAINING -----------
    model_name = f"JEPA_models_pt/JEPA_model_OPP_fold{args.fold}_seed{seed}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    bad_epochs = 0
    with open(f"logs/JEPA_training_OPP_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | SEED: {seed}\n")
        log_file.flush()
        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            total_samples = 0

            loop = tqdm(train_loader, desc = f"Epoch {epoch+1}/{num_epochs}")
            for _, (x_batch, _) in enumerate(loop):
                x_batch = x_batch.to(device, non_blocking = True)

                optimizer.zero_grad()
                _, targets, _, _, _, predictions = model(x_batch)
                loss = criterion(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm = 1.0)
                optimizer.step()

                momentum = m_start + (m_end - m_start) * (global_step / total_steps)
                model.update_target_encoder(momentum = momentum)
                global_step += 1

                bsize = x_batch.size(0)
                total_loss += loss.item() * bsize
                total_samples += bsize
                loop.set_postfix(loss = f"{total_loss/total_samples:.4f}")

            train_loss = total_loss / total_samples
            val_loss, emb_std, mean_cos = validate_model_PRETRAIN_JEPA(model, val_loader, device, criterion)

            epoch_time = time.time() - epoch_start
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "val_loss": float(val_loss),
                "emb_std": float(emb_std),
                "mean_cos": float(mean_cos)
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | emb_std: {emb_std:.4f} | mean_cos: {mean_cos:.4f} | epoch_time: {epoch_time:.2f}s")
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | emb_std: {emb_std:.4f} | mean_cos: {mean_cos:.4f} | epoch_time: {epoch_time:.2f}s\n")
            log_file.flush()

            if val_loss < best_val_loss - 1e-12:
                best_val_loss = float(val_loss)
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
            scheduler.step()

        if best_state is not None:
            torch.save(best_state, model_name)
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")