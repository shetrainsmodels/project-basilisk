from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from MambaSSL_JEPA_Model import MambaJEPA, MambaEncoderModel, PredictorTransformer
import numpy as np
import torch
import json
import os

@torch.no_grad()
def test_model(model, test_loader, device):
    '''
    Runs inference on the test set and computes evaluation metrics.
    '''
    model.eval()
    all_preds = []
    all_labels = []
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device, non_blocking = True), y_batch.to(device, non_blocking = True)
        
        logits_ = model(x_batch)
        predictions = logits_.argmax(dim = -1)

        all_preds.extend(predictions.numpy(force = True)) # move to CPU
        all_labels.extend(y_batch.numpy(force = True)) # move to CPU

    acc = accuracy_score(all_labels, all_preds)    
    report = classification_report(all_labels, all_preds, target_names = ["STAND", "WALK", "SIT", "LIE"], output_dict = True)
    f1 = f1_score(all_labels, all_preds, average = "macro") # macro: F1 avg equally among classes, for general performance
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return acc, report, f1, conf_matrix

def save_json(dataset, fold, seed, seed_result, out_dir = "results_json") -> None:
    '''
    Saves the results of a single seed run to a JSON file for the given fold.
    Results are stored under "runs" keyed by seed for easy lookup.
    '''
    os.makedirs(out_dir, exist_ok = True)
    json_path = os.path.join(out_dir, f"{dataset}_fold{fold}_results.json")
    # check if json file exists.
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_info = json.load(f)
    else:
        json_info = {
            "fold": int(fold),
            "runs": {}
        }
    # add seed results into the dict
    json_info["runs"][f"seed_{seed}"] = seed_result
    with open(json_path, "w") as f:
        json.dump(json_info, f, indent = 4)
    
            
        
    