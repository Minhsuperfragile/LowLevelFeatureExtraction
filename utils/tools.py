import torch
from model.models import SimpleNeuralNetwork, ResNetHybrid
from tqdm import tqdm
from utils.lowlevelfeatures import LowLevelFeatureExtractor
from utils.PFI import per_class_accuracy
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import multiprocessing
import time
from typing import *
from datetime import datetime
import pytz

def train_model(model: SimpleNeuralNetwork | ResNetHybrid, 
                train_loader: torch.utils.data.DataLoader, 
                epochs: int, 
                features_set: str,
                loss_path='loss_history.npy') -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_is_resnet = isinstance(model, ResNetHybrid)

    model.to(device)
    model.train()
    print(f"Training model using {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if not model_is_resnet:
        for epoch in (pbar:= tqdm(range(epochs))):
            model.train()
            running_loss = 0.0
            for inputs, metadata, labels in train_loader:
                optimizer.zero_grad()

                labels = labels.long().to(device)
                metadata = metadata.to(device)

                outputs = model(metadata)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            pbar.set_description(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset)}')

        print(f"Training on {features_set} complete!")
    else:
        loss_history=[]
        for epoch in tqdm(range(epochs), desc="Epochs"):  # Wrap epochs in tqdm
            model.train()
            running_loss = 0.0
            
            # Wrap the inner loop (batches) in tqdm
            for images, metadata, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                
                # Move data to the device (GPU/CPU)
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.long().to(device)
                
                # Forward pass
                outputs = model(images, metadata)
                
                # Compute loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader.dataset)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        np.save(loss_path,np.array(loss_history))
        print(f"Training on {features_set} complete!")

def evaluate_model(model: SimpleNeuralNetwork | ResNetHybrid, 
                   test_loader: torch.utils.data.DataLoader,  
                   features_set = str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_is_resnet = isinstance(model, ResNetHybrid)

    model.to(device)
    model.eval()
    predicteds, label = np.array([]),  np.array([])

    if not model_is_resnet:
        with torch.no_grad():
            for inputs, metadata, labels in test_loader:
                labels = labels.long().to(device)
                metadata = metadata.to(device)
                outputs = model(metadata)
                _, predicted = torch.max(outputs.data, 1)

                label = np.append(label, labels.cpu().numpy())
                predicteds = np.append(predicteds, predicted.cpu().numpy())
    else:
        with torch.no_grad():  # No gradients are needed for evaluation
            for images, metadata, labels in test_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

                outputs = model(images, metadata)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get the predicted class

                label = np.append(label, labels.cpu().numpy())
                predicteds = np.append(predicteds, preds.cpu().numpy())

    # Calculate accuracy
    accuracy = (predicteds == label).sum() / len(label) * 100
    cm = per_class_accuracy(predicteds, label)
    return accuracy, cm

class MultiprocessingExtractor:
    def __init__(self):
        pass

    def __extract_features(self, image_path: os.PathLike, root_folder: os.PathLike, llf: LowLevelFeatureExtractor, transform: transforms.Compose) -> np.ndarray:
        image_path = os.path.join(root_folder, image_path)
        image = transform(Image.open(image_path)) # Convert to gray-scale
        image = np.array(image)
        features = llf.process_single_image(image)
        return features

    # Function to process a smaller dataframe chunk
    def __process_chunk(self, df_chunk: pd.DataFrame, root_folder: os.PathLike, llf: LowLevelFeatureExtractor, transform: transforms.Compose):
        return [self.__extract_features(row['image'], root_folder, llf, transform) for _, row in df_chunk.iterrows()]

    def process_dataframe(
            self,
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
            results = pool.starmap(self.__process_chunk, [(chunk, root_folder, llf, transform) for chunk in df_chunks])

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

class FilesProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_csv_from_folder(root: os.PathLike, title: str, save_folder: os.PathLike= "./") -> dict:
        """
        Generate a csv that contain all path to images in the given folder
        Folder must be organized like this:
        - root folder 
            - train 
                - class 0
                - class 1 
                - ...
            - test 
                - class 0
                - class 1 
                - ...
        """
        class_map = {}
        image_path = []
        labels = []
        for folder in os.listdir(root):
            classes = os.listdir(os.path.join(root,folder))
            for class_index, class_name in enumerate(classes):
                images = os.listdir(os.path.join(folder, class_name))
                for img in images:
                    image_path.append(os.path.join(folder, class_name, img))
                    labels.append(class_index)

            class_map[class_index] = class_name

            df = pd.DataFrame(data={"image": image_path, "label_id": labels})
            df.to_csv(os.path.join(save_folder, f'{title}_{folder}.csv'), index = False)

        return class_map
    
    @staticmethod
    def get_highest_importance_features(text_file: os.PathLike, top_k:int = 20) -> pd.DataFrame:
        with open(text_file) as f:
            lines = f.readlines()
        
        overall_importance = []
        overall_name = []
        overall_id = []

        for line in lines:
            if len(line) == 0:
                continue # ignore empty lines
                
            elements = line.split(',')
            if len(elements) == 1:
                name = elements[0].strip()
                continue

            if len(elements) == 2:
                feat = elements[0].split('_')[-1]
                score = elements[1].strip()
            
            overall_importance.append(score)
            overall_name.append(name)
            overall_id.append(feat)

        df = pd.DataFrame(data={"name": overall_name, "id": overall_id, "score": overall_importance})
        df.sort_values(by="score", ascending=False, inplace=True)
        return df.head(top_k)

    @staticmethod
    def create_result_df(column_name: List[str], feature_name: List[str], n_class: int):
        """
        Create a result dataframe with different features set on different models
        Shape = [features sets, models, classes precision]
        column_name is a list of models
        feature_name is a list of features name which will be used as index
        n_class is number of classes to calculate precision
        """

        column_name.insert(0, 'features_name')

        data_ = {"features_name": feature_name}
        data_[tuple(column_name)] = pd.NA
        data_["features_name"] = feature_name
        df = pd.DataFrame(data_, columns=column_name)

        for i in range(n_class):
            data_ = {"features_name": [f"{x}_{i}" for x in feature_name]}
            data_[tuple(column_name)] = pd.NA
            df = pd.concat([df, pd.DataFrame(data_, columns=column_name)], ignore_index=True)

        df.set_index('features_name', inplace=True)
        return df
    
    @staticmethod
    def distill_features(top_k_features: pd.DataFrame, n_class: int) -> pd.DataFrame:
        """
        Distill features from a dataframe with top_k_features.
        """
        pass
    #TODO

def get_current_datetime(timezone='Asia/Ho_Chi_Minh'):
    """
    Returns the current date (Y-m-d) and time (H-M) in the specified timezone.
    
    :param timezone: str - the name of the timezone (default is 'Asia/Ho_Chi_Minh')
    """
    # Get the timezone object
    tz = pytz.timezone(timezone)
    
    # Get the current time in UTC and convert it to the specified timezone
    current_datetime = datetime.now(tz)
    
    # Return the formatted date and time
    return current_datetime.strftime("%Y-%m-%d_%H:%M")
