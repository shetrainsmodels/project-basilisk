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


    
            
        
    