import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

# Swish activation function as nn.Module
class Swish_Module(nn.Module):
    def __init__(self):
        super(Swish_Module, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# Hybrid model combining ResNet and Metadata features
class ResNetHybrid(nn.Module):
    def __init__(self, resnet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(ResNetHybrid, self).__init__()

        self.n_meta_features = n_meta_features

        # Load pre-trained ResNet model (ResNet50, ResNet101, etc.)
        self.resnet = getattr(models, resnet_type)(pretrained=pretrained)
        
        # Dropout layers for regularization
        self.dropout = nn.Dropout(0.5)

        # Get input features from ResNet classifier (Fully connected layer)
        in_ch = self.resnet.fc.in_features

        # If metadata features are provided, add a separate network to process metadata
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            # Increase the input channels by the metadata feature size
            in_ch += n_meta_dim[1]

        # Final classification layer
        self.myfc = nn.Linear(in_ch, out_dim)

        # Replace ResNet fully connected layer with identity to exclude it from forward pass
        self.resnet.fc = nn.Identity()

    def extract(self, x):
        # Forward pass through ResNet without the classifier part
        return self.resnet(x)

    def forward(self, x, x_meta=None):
        # Extract features from the image
        x = self.extract(x)

        # If metadata is available, process it through the metadata network
        if self.n_meta_features > 0 and x_meta is not None:
            x_meta = self.meta(x_meta)
            # Concatenate image features and metadata features
            x = torch.cat((x, x_meta), dim=1)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Final classification layer
        out = self.myfc(x)
        
        return out
