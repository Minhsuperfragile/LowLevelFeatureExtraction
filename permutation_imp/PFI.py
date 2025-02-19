import numpy as np
import torch
from sklearn.metrics import accuracy_score

def permutation_feature_importance(model, device, X, y, n_repeats=10, metric=accuracy_score):
    """
    Compute permutation feature importance from scratch.
    
    Args:
      - model (torch.nn.Module): Trained PyTorch model.
      - device (str): 'cpu' or 'cuda'.
      - X (torch.Tensor): Feature tensor (num_samples, num_features).
      - y (torch.Tensor): Ground truth labels.
      - n_repeats (int): Number of times to shuffle each feature.
      - metric (function): Scoring function, e.g., accuracy_score.
    
    Returns:
      - importance_scores (np.array): Importance scores for each feature.
    """

    model.eval()  # Ensure model is in evaluation mode
    X, y = X.to(device), y.to(device)  # Move to device

    # 1️⃣ Compute baseline accuracy
    with torch.no_grad():
        baseline_preds = model(X)
        baseline_preds = torch.argmax(baseline_preds, dim=1).cpu().numpy()  # Convert to labels
    baseline_score = metric(y.cpu().numpy(), baseline_preds)
    
    num_features = X.shape[1]
    importance_scores = np.zeros(num_features)

    # 2️⃣ Iterate over each feature
    for feature_idx in range(num_features):
        temp_scores = []
        
        for _ in range(n_repeats):
            X_shuffled = X.detach().clone()  # Clone safely without gradients
            shuffled_feature = X[:, feature_idx].clone()[torch.randperm(X.shape[0])]  # Shuffle column
            X_shuffled[:, feature_idx] = shuffled_feature  # Replace with shuffled values

            # 3️⃣ Compute new accuracy after shuffling
            with torch.no_grad():
                shuffled_preds = model(X_shuffled)
                shuffled_preds = torch.argmax(shuffled_preds, dim=1).cpu().numpy()
            shuffled_score = metric(y.cpu().numpy(), shuffled_preds)

            # Store the performance drop
            temp_scores.append(baseline_score - shuffled_score) 
        
        # 4️⃣ Average across repeats
        importance_scores[feature_idx] = np.mean(temp_scores)

    return importance_scores