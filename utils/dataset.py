import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class CSVMetadataDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Base directory for images
        self.transform = transform  # Transformations

        # Extract image paths, labels, and metadata
        # self.image_paths = self.data.iloc[:, 0].values  # Image paths
        self.labels = self.data.iloc[:, 1].values.astype(int)  # Labels
        self.metadata = self.data.iloc[:, 2:].values.astype(float)  # Metadata features (Numpy array)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)  # Convert metadata to tensor

        return np.empty(0), metadata, label # None is used for image data, metadata is used instead

class CSVImageMetadataDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None, img_size=(224, 224)):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Base directory for images
        self.img_size = img_size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(self.img_size),  # Resize images to a consistent size
            transforms.ToTensor()  # Convert the image to a tensor
        ])

        # Extract image paths, labels, and metadata
        self.image_paths = self.data.iloc[:, 0].values  # Image paths
        self.labels = self.data.iloc[:, 1].values.astype(int)  # Labels
        self.metadata = self.data.iloc[:, 2:].values.astype(float)  # Metadata features (Numpy array)
        
    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])  # Get full image path
        image = Image.open(img_path).convert("RGB")  # Load image
        label = self.labels[idx]  # Get label
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)  # Convert metadata to tensor

        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, metadata, label  # Return image, metadata, and label