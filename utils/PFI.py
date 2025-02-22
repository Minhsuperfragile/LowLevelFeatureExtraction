from typing import *
import pandas as pd
from sklearn.metrics import confusion_matrix
from model.models import SimpleNeuralNetwork
import numpy as np
import torch

def class_precision(predictions: np.ndarray, labels: np.ndarray, class_:int = 1):
    cm = confusion_matrix(labels, predictions)
    
    pass

def permutation_feature_importance(
        model: SimpleNeuralNetwork, 
        features: pd.DataFrame, 
        label: pd.DataFrame, 
        metric: Callable) -> None:
    model.eval()

    half_size = features.shape[0] // 2
    X1, X2 = features[:half_size], features[half_size:]
    y1 = label[:half_size]

    # Compute base metric
    with torch.no_grad():
        base_preds = model(X1)
        base_preds = torch.argmax(base_preds, dim=1).cpu().numpy()
    base_metric = metric(base_preds, y1)

    importance = []
    for features_idx in range(X1.shape[1]):
        X_swap = X1.clone()
        X_swap.iloc[:, features_idx ] = X2.iloc[:, features_idx]

        with torch.no_grad():
            swap_pred = model(X_swap)
            swap_pred = torch.argmax(swap_pred, dim=1).cpu().numpy()

        swap_metric = metric(swap_pred, y1)
        importance.append(swap_pred - base_metric)

    return importance