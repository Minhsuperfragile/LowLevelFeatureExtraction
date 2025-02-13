import pyfeats
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import torchvision

mask = None
param_list = [
{
    'function': pyfeats.fos, #First Order Statistics/Statistical Features (FOS/SF)
    'params': {'mask': mask },
    'features_set': ['features']
},
{
    "function": pyfeats.glcm_features, #Gray Level Co-occurence Matrix (GLCM/SGLDM)
    "params": {'ignore_zeros': True},
    "features_set": ['features_mean', 'features_range']
}, 
{
    "function": pyfeats.glds_features, #Gray Level Difference Statistics (GLDS)
    "params": {'mask': mask, 'Dx': [0,1,1,1], 'Dy': [1,1,0,-1]},
    "features_set": ['features']
},
{
    "function": pyfeats.ngtdm_features, #Neighborhood Gray Tone Difference Matrix (NGTDM)
    "params": {'mask': mask, 'd': 1},
    "features_set": ['features']
},
{
    "function": pyfeats.sfm_features, #Statistical Feature Matrix (SFM)
    'params': {'mask': mask, 'Lr':4, 'Lc':4},
    'features_set': ['features']
},
{
    "function": pyfeats.lte_measures, #Law's Texture Energy Measures (LTE/TEM)
    'params': {'mask': mask, 'l': 7},
    'features_set': ['features']
},
{
    "function": pyfeats.fdta, #Fractal Dimension Texture Analysis (FDTA)
    'params': {'mask': mask, 's':3},
    'features_set': ['features']
},
{
    "function": pyfeats.glrlm_features, #Gray Level Run Length Matrix (GLRLM)
    'params': {'mask': mask, 'Ng': 256},
    'features_set': ['features']
},
{
    "function": pyfeats.fps, #Fourier Power Spectrum (FPS)
    'params': {'mask': mask},
    'features_set': ['features']
},
{
    "function": pyfeats.shape_parameters, #Shape Parameters
    'params': {'mask': mask, 'perimeter': np.ones((128,128)), 'pixels_per_mm2':1},
    'features_set': ['features']
},
{
    "function": pyfeats.glszm_features, #Gray Level Size Zone Matrix (GLSZM)
    'params': {'mask': mask},
    'features_set': ['features']
},
{
    "function": pyfeats.hos_features, #Higher Order Spectra (HOS)
    'params': {'th': [135,140]},
    'features_set': ['features']
},
{
    "function": pyfeats.lbp_features, #Local Binary Pattern (LPB)
    'params': {'mask': mask,  'P':[8,16,24], 'R':[1,2,3]}, 
    'features_set': ['features']
},
{
    "function": pyfeats.grayscale_morphology_features, #Gray-scale Morphological Analysis
    'params': {"N": 30},
    'features_set': ['pdf', 'cdf']
},
# { Deprecated: generate NaN values in the output}
#     "function": pyfeats.multilevel_binary_morphology_features, #Multilevel Binary Morphological Analysis
#     'params': {'mask': mask, 'N': 30, 'thresholds': [25, 50]},
#     'features_set': ['pdf_L', 'pdf_M', 'pdf_H', 'cdf_L', 'cdf_M', 'cdf_H']
# },
{
    'function': pyfeats.histogram, #Histogram
    'params': {'mask': mask, 'bins': 32},
    'features_set': ['H']
},
{
    "function": pyfeats.multiregion_histogram, #Multi-region histogram
    'params': {'mask': mask, 'bins': 32, 'num_eros': 3, 'square_size': 3},
    'features_set': ['features']
},
# { Deprecated: generate all 0 values in the output}
#     "function": pyfeats.correlogram, #Correlogram
#     'params': {'mask': mask, 'bins_digitize' : 32, 'bins_hist' : 32, 'flatten' : True},
#     'features_set': ['Hd', 'Ht']
# },
{
    "function": pyfeats.fdta, #Fractal Dimension Texture Analysis (FDTA)
    'params': {'mask': mask, 's': 3},
    'features_set': ['h']
},
{
    "function": pyfeats.amfm_features, #Amplitude Modulation – Frequency Modulation (AM-FM)
    'params': {'bins': 32},
    'features_set': ['features'] # Take long time to calculate
},
{
    "function": pyfeats.dwt_features, #Discrete Wavelet Transform (DWT)
    'params': {'mask': mask, 'wavelet': 'bior3.3', 'levels': 3},
    'features_set': ['features']
},
{
    "function": pyfeats.swt_features, #Stationary Wavelet Transform (SWT)
    'params': {'mask': mask, 'wavelet': 'bior3.3', 'levels': 3},
    'features_set': ['features']
},
{
    "function": pyfeats.wp_features, #Wavelet Packets (WP)
    'params': {'mask': mask, 'wavelet': 'coif1', 'maxlevel': 3},
    'features_set': ['features']
},
{
    "function": pyfeats.gt_features, #Gabor Transform (GT)
    'params': {'mask': mask, 'deg': 4, 'freq': [0.05, 0.4]},
    'features_set': ['features']
},
{
    "function": pyfeats.zernikes_moments, #Zernikes’ Moments
    'params': {'radius': 9},
    'features_set': ['features']
},
{
    "function": pyfeats.hu_moments, #Hu’s Moments
    'params': {},
    'features_set': ['features']
},
{
    "function": pyfeats.tas_features, #Threshold Adjacency Matrix (TAS)
    'params': {},
    'features_set': ['features']
},
{
    "function": pyfeats.hog_features, #Histogram of Oriented Gradients (HOG)
    'params': {'ppc': 8, 'cpb': 3},
    'features_set': ['features']
}
]

class ToNumpy():
    def __call__(self, *args, **kwds):
        return np.array(args[0])

class LowLevelFeatureExtractor:
    def __init__(self, function: Callable, # pyfeats function to call
                 params: Dict[str, Any] = None, # pyfeats function parameters
                 features_set: List[str] = None, # list of features to extract
                 ) -> None:
        self.function = function
        self.params = params if params is not None else {}
        self.features_set = features_set

    def __call__(self, images):
        images = np.array(images) if not isinstance(images, np.ndarray) else images
        features = []

        for image in images:
            image= np.squeeze(image)

            features_output = self.function(image, **self.params)
            
            features_set = {feature: value for feature, value in zip(self.features_set, features_output)}

            features_set = np.concatenate([features_set[key] for key in features_set.keys()], axis=0)
            # features.append(np.expand_dims(features_set, axis=0))
            features.append(features_set)

        self.features_size = features_set.shape[0]

        return np.stack(features, axis=0)
    
    def get_features_size(self) -> int:
        sample_image = np.random.randint(0, 256, (32, 32))
        features_set = self(np.expand_dims(sample_image, axis=0))

        self.features_size = features_set.shape[1]

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

class SwishActivation(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self, inputs: int, classes = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(inputs + 6, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.swish = SwishActivation()
        self.batchNorm = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, metadata):
        x = torch.cat((x, metadata), dim=1).float()
        x = self.fc1(x)
        x = self.batchNorm(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x = self.softmax(x)
        return x

def train_model(model: SimpleNeuralNetwork, train_loader: torch.utils.data.DataLoader, llf: LowLevelFeatureExtractor, epochs: int, features_set: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"Training model using {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, metadata, labels in train_loader:
            optimizer.zero_grad()
            inputs = torch.Tensor(llf(inputs)).to(device)
            labels = labels.long().to(device)
            metadata = metadata.to(device)

            outputs = model(inputs, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset)}')

    print(f"Training on {features_set} complete!")

def evaluate_model(model: SimpleNeuralNetwork, test_loader: torch.utils.data.DataLoader, llf: LowLevelFeatureExtractor, features_set = str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, metadata, labels in test_loader:
            inputs = torch.Tensor(llf(inputs)).to(device)
            labels = labels.long().to(device)
            metadata = metadata.to(device)
            outputs = model(inputs, metadata)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted.to(device) == labels).sum().item()

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
    print(features)