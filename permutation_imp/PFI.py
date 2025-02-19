import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def permutation_feature_importance(model, device, X, y, n_repeats=10, metric=accuracy_score):
    """
    Compute permutation feature importance separately for each half of the dataset.
    """
    model.eval()
    X, y = X.to(device), y.to(device)

    # Compute baseline accuracy
    with torch.no_grad():
        baseline_preds = model(X)
        baseline_preds = torch.argmax(baseline_preds, dim=1).cpu().numpy()
    baseline_score = metric(y.cpu().numpy(), baseline_preds)

    num_features = X.shape[1]
    importance_scores = np.zeros(num_features)

    # Split data in half
    half_size = X.shape[0] // 2
    X1, X2 = X[:half_size], X[half_size:]
    y1, y2 = y[:half_size], y[half_size:]

    importance_scores_1 = np.zeros(num_features)
    importance_scores_2 = np.zeros(num_features)

    for feature_idx in range(num_features):
        temp_scores_1 = []
        temp_scores_2 = []

        for _ in range(n_repeats):
            # Shuffle feature in first half
            X1_shuffled = X1.clone()
            shuffled_feature_1 = X1[:, feature_idx][torch.randperm(X1.shape[0])]
            X1_shuffled[:, feature_idx] = shuffled_feature_1

            # Shuffle feature in second half
            X2_shuffled = X2.clone()
            shuffled_feature_2 = X2[:, feature_idx][torch.randperm(X2.shape[0])]
            X2_shuffled[:, feature_idx] = shuffled_feature_2

            # Compute accuracy after shuffling
            with torch.no_grad():
                shuffled_preds_1 = model(X1_shuffled)
                shuffled_preds_1 = torch.argmax(shuffled_preds_1, dim=1).cpu().numpy()
                shuffled_score_1 = metric(y1.cpu().numpy(), shuffled_preds_1)

                shuffled_preds_2 = model(X2_shuffled)
                shuffled_preds_2 = torch.argmax(shuffled_preds_2, dim=1).cpu().numpy()
                shuffled_score_2 = metric(y2.cpu().numpy(), shuffled_preds_2)

            temp_scores_1.append(baseline_score - shuffled_score_1)
            temp_scores_2.append(baseline_score - shuffled_score_2)

        # Store the average drop in performance
        importance_scores_1[feature_idx] = np.mean(temp_scores_1)
        importance_scores_2[feature_idx] = np.mean(temp_scores_2)

    # Average the results from both halves
    importance_scores = (importance_scores_1 + importance_scores_2) / 2

    return importance_scores
