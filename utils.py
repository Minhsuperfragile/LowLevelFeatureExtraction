import pyfeats
from typing import *
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
from tabulate import tabulate
import multiprocessing
import time
from sklearn.metrics import confusion_matrix, accuracy_score

#region Feature Specifications
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
# { Deprecated: generate all 0 values in the output
#     'function': pyfeats.histogram, #Histogram
#     'params': {'mask': mask, 'bins': 32},
#     'features_set': ['H']
# },
# { Deprecated: generate all 0 values in the output
#     "function": pyfeats.multiregion_histogram, #Multi-region histogram
#     'params': {'mask': mask, 'bins': 32, 'num_eros': 3, 'square_size': 3},
#     'features_set': ['features']
# },
# { Deprecated: generate all 0 values in the output
#     "function": pyfeats.correlogram, #Correlogram
#     'params': {'mask': mask, 'bins_digitize' : 32, 'bins_hist' : 32, 'flatten' : True},
#     'features_set': ['Hd', 'Ht']
# },
{
    "function": pyfeats.fdta, #Fractal Dimension Texture Analysis (FDTA)
    'params': {'mask': mask, 's': 3},
    'features_set': ['h']
},
# { Deprecated: Too long to calculate (83 hours)
#     "function": pyfeats.amfm_features, #Amplitude Modulation – Frequency Modulation (AM-FM)
#     'params': {'bins': 32},
#     'features_set': ['features'] # Take long time to calculate
# },
# {  Deprecated: generate NaN values in the output
#     "function": pyfeats.dwt_features, #Discrete Wavelet Transform (DWT)
#     'params': {'mask': mask, 'wavelet': 'bior3.3', 'levels': 3},
#     'features_set': ['features']
# },
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
# { Deprecated: features size depends on the image size and ppc/cpb values. 
#     "function": pyfeats.hog_features, #Histogram of Oriented Gradients (HOG)
#     'params': {'ppc': 8, 'cpb': 3},
#     'features_set': ['features']
# }
]
#endregion

#region Model Definition
class ToNumpy():
    def __call__(self, *args, **kwds):
        return np.array(args[0])

class LowLevelFeatureExtractor:
    def __init__(self, function: Callable, # pyfeats function to call
                 params: Dict[str, Any] = None, # pyfeats function parameters
                 features_set: List[str] = None, # list of features to extract,
                 image_size: Tuple[int, int] = None) -> None: # size of the input image 
        self.function = function
        self.params = params if params is not None else {}
        self.features_set = features_set
        self.image_size = image_size if image_size is not None else (384,384)

    def __call__(self, images):
        images = np.array(images) if not isinstance(images, np.ndarray) else images
        features = []

        for image in images:
            image= np.squeeze(image)

            features_output = self.function(image, **self.params)
            
            features_set = {feature: value for feature, value in zip(self.features_set, features_output)}

            features_set = np.concatenate([features_set[key] for key in features_set.keys()], axis=0)

            features.append(features_set)

        return np.stack(features, axis=0)
    
    def process_single_image(self, image:np.ndarray) -> np.ndarray:
        features = self.function(image, **self.params)
        features = {feature: value for feature, value in zip(self.features_set, features)}
        features = np.concatenate([features[key] for key in features.keys()], axis=0)
        return features

    def get_features_size(self) -> int:
        sample_image = np.random.randint(0, 256, self.image_size).astype("uint8")
        features_set = self.process_single_image(sample_image)

        self.features_size = features_set.shape[0]

        return self.features_size

class CSVMetadataDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None, md=True):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Base directory for images
        self.transform = transform  # Transformations

        # Extract image paths, labels, and metadata
        # self.image_paths = self.data.iloc[:, 0].values  # Image paths
        self.labels = self.data.iloc[:, 1].values.astype(int)  # Labels
        mark = 2 if md else 8 # Index of the first metadata column
        self.metadata = self.data.iloc[:, mark:].values.astype(float)  # Metadata features (Numpy array)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)  # Convert metadata to tensor

        return np.empty(0), metadata, label # None is used for image data, metadata is used instead
    
    def get_input_size(self):
        return self.metadata.shape[1]

class CSVImageMetadataDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file
        self.root_dir = root_dir  # Base directory for images
        self.transform = transform  # Transformations

        # Extract image paths, labels, and metadata
        self.image_paths = self.data.iloc[:, 0].values  # Image paths
        self.labels = self.data.iloc[:, 1].values.astype(int)  # Labels
        self.metadata = self.data.iloc[:, 8:].values.astype(float)  # Metadata features (Numpy array)

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
    def __init__(self, inputs: int, classes:int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = torch.nn.Linear(inputs, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, classes)
        self.softmax = torch.nn.Softmax(dim=1)
        self.swish = SwishActivation()
        self.batchNorm = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, metadata = NotImplemented):
        # x = torch.cat((x, metadata), dim=1).float()
        x = self.fc1(x)
        x = self.batchNorm(x)
        x = self.sigmoid(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x = self.softmax(x)
        return x
#endregion

#region Model Training Function
def train_model(model: SimpleNeuralNetwork, train_loader: torch.utils.data.DataLoader, llf: LowLevelFeatureExtractor, epochs: int, features_set: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"Training model using {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in (pbar:= tqdm(range(epochs))):
        model.train()
        running_loss = 0.0
        for inputs, metadata, labels in train_loader:
            optimizer.zero_grad()
            # inputs = torch.Tensor(llf(inputs)).to(device)
            labels = labels.long().to(device)
            metadata = metadata.to(device)

            outputs = model(metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        pbar.set_description(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset)}')

    print(f"Training on {features_set} complete!")

def evaluate_model(model: SimpleNeuralNetwork, test_loader: torch.utils.data.DataLoader, llf: LowLevelFeatureExtractor, features_set = str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_predicted = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, metadata, labels in test_loader:
            # inputs = torch.Tensor(llf(inputs)).to(device)
            labels = labels.long().to(device)
            metadata = metadata.to(device)

            outputs = model(metadata)
            _, predicted = torch.max(outputs.data, 1)

            # Collect labels and predictions for confusion matrix
            all_labels.append(labels.cpu().numpy())
            all_predicted.append(predicted.cpu().numpy())

            # Update correct and total counts for overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Convert lists to numpy arrays
    all_labels = np.concatenate(all_labels)
    all_predicted = np.concatenate(all_predicted)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)

    # Calculate overall accuracy
    accuracy = 100 * correct / total

    # Print overall accuracy
    print(f"Overall Accuracy: {accuracy:.2f}%")

    l = []
    # Calculate and print accuracy for each class
    for i in range(cm.shape[0]):
        class_accuracy = cm[i, i] / cm[i].sum() * 100
        l.append(class_accuracy)
        print(f"Class {i} Accuracy: {class_accuracy:.2f}%")
    
    return accuracy, l
#endregion

#region Data Processing Functions
# Function to process an image path
def extract_features(image_path: os.PathLike, root_folder: os.PathLike, llf: LowLevelFeatureExtractor, transform: transforms.Compose) -> np.ndarray:
    image_path = os.path.join(root_folder, image_path)
    image = transform(Image.open(image_path)) # Convert to gray-scale
    image = np.array(image)
    features = llf.process_single_image(image)
    return features

# Function to process a smaller dataframe chunk
def process_chunk(df_chunk: pd.DataFrame, root_folder: os.PathLike, llf: LowLevelFeatureExtractor, transform: transforms.Compose):
    return [extract_features(row['image'], root_folder, llf, transform) for _, row in df_chunk.iterrows()]

def process_dataframe(
        df: pd.DataFrame, 
        llf: LowLevelFeatureExtractor ,
        transform: transforms.Compose,
        root_folder: os.PathLike = "../skin_data", # root image folder 
        save_path: os.PathLike = "./data/result.csv",
        n_process: int = None, 
        time_logger=True
        ) -> None:
    
    n_process = multiprocessing.cpu_count()//2 if n_process is None else n_process
    n_features = llf.get_features_size()
    start_time = time.time() if time_logger else 0
    df_chunks = np.array_split(df, n_process)

    # Start multiprocessing pool
    with multiprocessing.Pool(processes=n_process) as pool:
        results = pool.starmap(process_chunk, [(chunk, root_folder, llf, transform) for chunk in df_chunks])

    # Combine all processed results into a DataFrame
    flat_results = [item for sublist in results for item in sublist]  # Flatten list of lists\
    df_features = pd.DataFrame(flat_results, columns=[f'feature_{i}' for i in range(n_features)])
    df_features = pd.concat([df_features, df['image']], axis=1)

    # print(tabulate(df_features, headers = 'keys', tablefmt = 'psql'))

    # Merge features with original DataFrame
    df_final = df.merge(df_features, on="image")

    # Save dataframe to csv
    save_folder = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    df_final.to_csv(save_path, index=False)

    # Log the execution time
    # End time tracking
    if time_logger:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"✅ Extracted {save_path} in {execution_time:.2f} seconds.")

#endregion

#region Permutation Importance Calculation

def permutation_feature_importance(model: SimpleNeuralNetwork, df:pd.DataFrame ):
    test_df = df[:len(df)//2]
    swap_df = df[len(df)//2:]

    n_features = test_df.shape[1]

    # Compute base acc
    with torch.no_grad():
        base_predictions = model(torch.Tensor(test_df.iloc[:,1:].to_numpy()))
        _, base_predictions = torch.max(base_predictions.data, 1)
        base_accuracy = accuracy_score(test_df.iloc[:,0].to_numpy(), base_predictions)

    # Initialize a list to store the importance scores
    importances = []
    for i in range(7,n_features):
        # Create a copy of the test data and swap the feature column with value from different swap dataframe
        swapped_test_df = test_df.copy()
        swapped_test_df.iloc[:, i] = swap_df.iloc[:,i]
        # Calculate the model's prediction on the swapped test data
        with torch.no_grad():
            predictions = model(torch.Tensor(swapped_test_df.iloc[:,1:].to_numpy()))
            _, predictions = torch.max(predictions.data, 1)
            accuracy = accuracy_score(test_df.iloc[:,0].to_numpy(), predictions)
            importances.append(accuracy - base_accuracy)

    return importances