import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import argparse
import torch 
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from model import *
from utils.lowlevelfeatures import *

def plot_confusion_matrix(model, test_loader, device, class_names=None, save_path="confusion_matrix.png"):
    """
    Plots and saves the confusion matrix for a given PyTorch model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.
        class_names (list of str, optional): List of class names for labeling the matrix.
        save_path (str, optional): The path to save the confusion matrix plot.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients are needed for evaluation
        for images, metadata, labels in test_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

            outputs = model(images, metadata)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory

    print(f"Confusion matrix saved to {save_path}")
path='../data/vip'
root_path="../../skin_data"
test_csv=f'../data/vip/vaynen_train_vip_new.csv'
batch_size = 32
test_dataset = CSVImageMetadataDataset(csv_file=test_csv, root_dir=root_path)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model=torch.load('model_vip_new_15.pth')
plot_confusion_matrix(model,test_dataloader,device='cuda',class_names=['general','puscular'])


