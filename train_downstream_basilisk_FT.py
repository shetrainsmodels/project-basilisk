import os, sys
# make the shared data/ package (one level up, in MASKING/) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MambaSSL_JEPA_Model import HARMambaConfig, MambaDownstreamClassifier
from data.OPPORTUNITY_data import load_OPP_loco_data, data_split_OPP, make_loaders_OPP
from test_mamba import test_model, save_json 
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
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

print(f"Mamba version: {mamba_ssm.__version__}")
print(f"Path: {mm.__file__}")

parser = argparse.ArgumentParser(description = "basilisk_finetune")
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--fold", type = int, required = True)
parser.add_argument("--lam", type = float, required = True)
parser.add_argument("--w_per_class", type = str, default = "all")
args = parser.parse_args()
RUN = f"lam{args.lam:g}"                                                    # pretrained-encoder folder (shared by all label fractions)
OUT = f"{RUN}_lpft" if args.w_per_class == "all" else f"{RUN}_lpft_n{int(args.w_per_class)}"   # output tag: LP-FT results/logs/.pt (encoder still loaded from RUN)
os.makedirs(f"JEPA_models_pt/{OUT}", exist_ok = True)
os.makedirs("logs", exist_ok = True)
if args.dataset == "OPP":
    if args.fold in [1, 2, 3, 4]:
        training_files, validation_files, test_files = data_split_OPP(args.fold)
    else:
        raise ValueError(f"Fold must be 1, 2, 3 or 4. Got {args.fold}")
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

X_windows, y_windows, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows = load_OPP_loco_data(training_files, validation_files, test_files, verbose = True, drill = False)

def load_pretrained_encoder(model, device, fold, seed):
    checkpoint_path = f"JEPA_models_pt/{RUN}/JEPA_model_OPP_fold{fold}_seed{seed}.pt"

    state = torch.load(checkpoint_path, map_location = device, weights_only=True)
    encoder_state = {k.replace("target_encoder.", "", 1): v for k,v in state.items() if k.startswith("target_encoder.")}
    model.encoder.load_state_dict(encoder_state, strict = True)
    if "pe_target" in state:
        model.pe = state["pe_target"].to(device)
    print("JEPA weights loaded with PE:", model.pe is not None)
#  ----------------------------------------------------- VALIDATION -----------------------------------------------------
@torch.no_grad()
def validate_model(model, val_loader, device, criterion):
    '''
    Validation: avg loss per window and accuracy per window.
    '''
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)
        logits_ = model(x_batch)                                               # forward Pass
        loss = criterion(logits_, y_batch)                                     # computes mean batch loss

        bsize = y_batch.size(0)
        total_loss += loss.item() * bsize                                      # Total loss contribution of this batch

        predictions = logits_.argmax(dim = -1)                                 # gets the predicted class for each window
        total_correct += (predictions == y_batch).sum().item()                 # counts correct predictions
        total_samples += bsize

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names = ["STAND", "WALK", "SIT", "LIE"])
    return total_loss / total_samples, total_correct / total_samples, report
# --------------------------------
@torch.no_grad()
def embed_and_rank(model, loader, device, tag):
    model.eval()

    Z = []
    # --------------------------------------------------
    # 1. Extract one embedding per IMU window
    # --------------------------------------------------
    for x_batch, _ in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        # Raw IMU -> tokens
        t = model.encoder.tokenize(x_batch)

        # Add positional embedding if used
        if model.pe is not None:
            if model.pe.ndim == 3:
                pe = model.pe[:, :t.shape[1]]
            else:
                pe = model.pe[:t.shape[1]]

            t = t + pe.to(device=t.device, dtype=t.dtype)

        # Mamba encoder
        h = model.encoder.encode_tokens(t)

        # Mean pool over sequence/time dimension
        # [B, T, D] -> [B, D]
        z = h.mean(dim=1)

        Z.append(z.float().cpu())
    # --------------------------------------------------
    # 2. Basic checks
    # --------------------------------------------------
    if not Z:
        raise ValueError("Loader is empty.")

    Z = torch.cat(Z, dim=0)

    if len(Z) < 2:
        raise ValueError("At least two windows are required.")

    # --------------------------------------------------
    # 3. Center embeddings
    # --------------------------------------------------
    Zc = Z - Z.mean(dim=0, keepdim=True)
    # --------------------------------------------------
    # 4. Singular values / variance spectrum
    # --------------------------------------------------
    sv = torch.linalg.svdvals(Zc)

    # Variance associated with each singular direction
    energy = sv.square()
    total_energy = energy.sum()

    if total_energy.item() <= 1e-12:
        # All embeddings are effectively identical
        eff = 0.0
        d95 = 0
        d99 = 0

    else:
        # Fraction of representation variance
        # explained by each singular direction
        p = energy / total_energy

        # Effective rank
        q = sv / (sv.sum() + 1e-12)
        entropy = -(q * q.clamp_min(1e-12).log()).sum()
        eff = torch.exp(entropy).item()

        # Cumulative explained variance
        ev = torch.cumsum(p, dim=0)

        # Number of dimensions required for
        # 95% and 99% of variance
        d95 = int((ev < 0.95).sum().item()) + 1
        d99 = int((ev < 0.99).sum().item()) + 1

    # --------------------------------------------------
    # 5. Mean cosine similarity between different windows
    # --------------------------------------------------
    Zn = F.normalize(Z, dim=1, eps=1e-12)

    n = len(Zn)

    # Equivalent to computing all pairwise cosine
    # similarities, but WITHOUT creating an NxN matrix.
    sum_vec = Zn.sum(dim=0)

    all_dot_products = sum_vec.square().sum()
    self_dot_products = Zn.square().sum()

    mcos = (
        (all_dot_products - self_dot_products)
        / (n * (n - 1))
    ).item()

    # --------------------------------------------------
    # 6. Print diagnostics
    # --------------------------------------------------
    print(
        f"[RANK:{tag}] "
        f"N={Z.shape[0]} "
        f"D={Z.shape[1]} "
        f"eff_rank={eff:.2f} "
        f"d95={d95} "
        f"d99={d99} "
        f"mean_cos={mcos:.4f}"
    )
    return eff
#  ----------------------------------------------------------------------------------------------------------------------
#  ---------------------------------------------------- LOSO TRAINING ---------------------------------------------------
def subsample_per_class(X_windows, y_windows, w_per_class, seed):
    rng = np.random.default_rng(seed)   # own RNG: depends only on seed -> same windows for every encoder
    keep = []
    for class_ in np.unique(y_windows):
        idx = np.where(y_windows == class_)[0]
        idx = rng.permutation(idx) # take the indices of each class and shuffle them
        keep.append(idx[:int(w_per_class)]) # keep only the first w_per_class indices of each class
    keep = np.sort(np.concatenate(keep, axis = 0))
    return X_windows[keep], y_windows[keep]

for seed in [42, 58, 7, 128, 92]: 
    g = set_seed(seed)
    if args.w_per_class == "all":
        X_tr, y_tr = X_windows, y_windows
    else:
        X_tr, y_tr = subsample_per_class(X_windows, y_windows, args.w_per_class, seed)
        print(f"[low-label] w_per_class = {args.w_per_class} | train windows: {len(y_tr)} | class counts: {dict(Counter(y_tr.tolist()))}")
    train_loader, val_loader, test_loader, label_encoder = make_loaders_OPP(X_tr, y_tr, X_validation_windows, y_validation_windows, X_test_windows, y_test_windows, generator = g, verbose = True)
    
    #  ----------- TRAINING SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(len(label_encoder.classes_))
    config = HARMambaConfig()
    model = MambaDownstreamClassifier(config, num_classes)
    model.to(device, non_blocking = True)
    num_epochs = 60
    patience = 5          # early-stopping patience 10 -> 5
    lr = 2e-4 
    # ---- LP-FT settings ----
    lp_epochs = 5         # phase 1: encoder frozen, train head only for lp_epochs, then unfreeze (LP-FT, Kumar 2022)
    enc_lr   = 1e-5       # encoder LR in phase 2 (unchanged)
    head_lr  = 1e-4       # head LR 3e-4 -> 1e-4
    wd       = 1e-2       # weight decay 1e-4 -> 1e-2
    
    # LOAD PRETRAINED ENCODER
    embed_and_rank(model, test_loader, device, f"randinit_f{args.fold}_s{seed}")
    load_pretrained_encoder(model, device, args.fold, seed)
    # LP-FT phase 1: encoder FROZEN, only the head trains (unfrozen at epoch lp_epochs inside the loop)
    for param in model.encoder.parameters():
        param.requires_grad = False
    embed_and_rank(model, test_loader, device, f"JEPA_f{args.fold}_s{seed}")
        
    #WCE
    #y_train_encoded = label_encoder.transform(np.asarray(y_windows))
    #counts = np.bincount(y_train_encoded, minlength = num_classes)
    #weights = torch.tensor(1.0 / counts, dtype=torch.float32, device = device)
    #criterion = nn.CrossEntropyLoss(weight = weights)
    criterion = nn.CrossEntropyLoss()
    # LP-FT phase 1 optimizer: head only
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr = head_lr, weight_decay = wd)
 #  ----------- TRAINING -----------
    model_name = f"JEPA_models_pt/{OUT}/model_JEPA_CLA_OPP_fold{args.fold}_seed{seed}.pt"
    epoch_history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = None
    best_state = None
    bad_epochs = 0
    prof_out = None
    with open(f"logs/model_JEPA_CLA_OPP_{OUT}_fold{args.fold}.txt", "a") as log_file:
        log_file.write(f"\nTRAINING STARTING AT: {datetime.now()}\n")
        log_file.write(f"Model: {model_name} | SEED: {seed}\n")
        log_file.flush()
        for epoch in range(num_epochs):
            if epoch == lp_epochs:
                # LP-FT phase 2: unfreeze encoder, discriminative LRs; restart best/patience so the saved checkpoint is a fine-tuned one
                for param in model.encoder.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": enc_lr}, {"params": model.classifier.parameters(), "lr": head_lr}], weight_decay = wd)
                best_val_loss = float("inf"); best_val_acc = 0.0; best_epoch = None; best_state = None; bad_epochs = 0
                print(f"[LP-FT] epoch {epoch+1}: encoder UNFROZEN (enc_lr={enc_lr}, head_lr={head_lr}, wd={wd})")
                log_file.write(f"[LP-FT] epoch {epoch+1}: encoder UNFROZEN (enc_lr={enc_lr}, head_lr={head_lr}, wd={wd})\n")
            epoch_start = time.time() # START EPOCH TIME
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
    
            loop = tqdm(train_loader, desc= f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (x_batch, y_batch) in enumerate(loop):
                x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)    
                optimizer.zero_grad()                                             # clear previous gradients
                logits_ = model(x_batch)                                          # forward pass
                loss = criterion(logits_, y_batch)                                # computes mean batch loss
                loss.backward()                                                   # backward pass: compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
                optimizer.step()                                                  # update model weights
                bsize = y_batch.size(0)
                total_loss += loss.item() * bsize                                 # Total loss contribution of this batch
                
                predictions = logits_.argmax(dim = -1)                            # gets the predicted class for each window: picks the highest score in the last dimension (one highest score per window among the 4 classes)
                total_correct += (predictions == y_batch).sum().item()            # number of correct predictions
                total_samples += bsize
                loop.set_postfix(loss= f"{total_loss/total_samples:.4f}", acc=f"{total_correct/total_samples:.4f}")
    
            train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
            val_loss, val_acc, report = validate_model(model, val_loader, device, criterion)

            epoch_time = time.time() - epoch_start # END EPOCH TIME
            epoch_history.append({
                "epoch": epoch + 1,
                "tr_loss": float(train_loss),
                "tr_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc)                
            })
            print(f"\nEpoch: {epoch+1}/{num_epochs} | tr_Loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s") 
            log_file.write(f"Epoch: {epoch+1}/{num_epochs} | tr_loss: {train_loss:.4f} | tr_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | epoch_time: {epoch_time:.2f}s\n")
            log_file.flush()
                
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = float(val_loss)
                best_val_acc = float(val_acc)
                best_epoch = epoch + 1                   
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience and epoch >= lp_epochs:   # early stopping only in the fine-tuning phase
                    print(f"\nEarly stopping at epoch {epoch + 1} | Best Validation Loss: {best_val_loss:.4f}")
                    break
                    
        # Load Best Model !            
        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, model_name)
            # Run val on best model to generate report
            _, _, report = validate_model(model, val_loader, device, criterion) 
            print(report)
            log_file.write(f"\nBest Model Validation Report: \n{str(report)}")        
        log_file.write(f"TRAINING ENDING AT: {datetime.now()}\n")                           
        
        #  ----------- TEST -----------         
        acc, report, f1, conf_matrix = test_model(model, test_loader, device)
        seed_result = {
            "seed": int(seed),
            "fold": int(args.fold),
            "history": epoch_history,
            "summary": {
                "lam": float(args.lam),
                "mode": "lpft",
                "lp_epochs": lp_epochs, "enc_lr": enc_lr, "head_lr": head_lr, "weight_decay": wd, "patience": patience,
                "w_per_class": args.w_per_class,
                "n_train_windows": int(len(y_tr)),
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "best_val_acc": float(best_val_acc),
                "test_accuracy": float(acc),
                "test_report": report,
                "test_f1": float(f1),
                "test_conf_matrix": conf_matrix.tolist()               
            }
        }
        save_json(args.dataset, args.fold, seed, seed_result, out_dir = f"results_json/{OUT}")
        print(f"{'-'*90}")
        print(f"Test Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
        log_file.write(f"\nTest Results:\n Accuracy: {acc}\n Report:\n {report}\n F1: {f1}\n Confusion Matrix:\n {conf_matrix}")
#  ----------------------------------------------------------------------------------------------------------------------
    

