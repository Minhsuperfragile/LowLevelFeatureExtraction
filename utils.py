import pyfeats
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class LowLevelFeatureExtractor:
    def __init__(self, function: Callable, # pyfeats function to call
                 params: Dict[str, Any] = None, # pyfeats function parameters
                 features_set: List[str] = None, # list of features to extract
                 ) -> None:
        self.function = function
        self.params = params if params is not None else {}
        self.features_set = features_set

    def __call__(self, image):
        self.params['f'] = np.array(image)

        features_output = self.function(**self.params)
        
        features_set = {feature: value for feature, value in zip(self.features_set, features_output)}

        features_set = np.concatenate([features_set[key] for key in features_set.keys()], axis=0)

        self.features_size = features_set.shape[0]

        return features_set
    
    def get_features_size(self) -> int:
        sample_image = np.random.randint(0, 256, (32, 32))
        features_set = self(sample_image)
        self.features_size = features_set.shape[0]

        return self.features_size

class CSVImageMetadataDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Base directory for images
        self.transform = transform  # Transformations

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


class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, inputs: int, classes = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(inputs + 6, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, metadata):
        x = torch.cat((x, metadata), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(model: SimpleNeuralNetwork, train_loader: torch.utils.data.DataLoader, epochs: int, features_set: str):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset)}')

    print(f"Training on {features_set} complete!")

def evaluate_model(model: SimpleNeuralNetwork, test_loader: torch.utils.data.DataLoader, features_set = str):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return f"Accuracy: {100 * correct / total}% On {features_set}"

if __name__ == "__main__":
    # Example usage
    import os, cv2

    path = "C:\\Users\\trong\\Documents\\skin_data\\train\\12"
    image_path = os.path.join(path, os.listdir(path)[0])

    # Read image from file
    image = cv2.imread(image_path)

    # Convert image to gray scale using OpenCV function
    gray_scale_image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an instance of LowLevelFeatureExtractor
    llf = LowLevelFeatureExtractor(function=pyfeats.glcm_features, 
                               params={'ignore_zeros': True}, 
                               features_set=['features_mean', 'features_range'])
    
    # Call the extractor with the gray scale image
    features = llf(gray_scale_image)
    print(llf.features_size)