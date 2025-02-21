import os
import argparse
import torch 
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from model import *
from utils.lowlevelfeatures import *
#kiem tra thiet bi

def get_transforms(target_size=224):
    """
    Transform pipeline cho các ảnh có kích thước khác nhau:
    - Resize ảnh sao cho kích thước nhỏ nhất đạt target_size
    - Cắt trung tâm về kích thước target_size x target_size
    - Convert sang tensor và normalize (theo thông số của ImageNet)
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])



def train_model(model, train_loader, epochs: int, features_set: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    print(f"Training model using {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    print(f"Training on {features_set} complete!")



def evaluate_model(model, test_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode

    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # No gradients are needed for evaluation
        for images, metadata, labels in test_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)

            outputs = model(images, metadata)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

    # Calculate accuracy
    accuracy = correct_preds / total_preds
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy


path='/mnt/e/VAST/Low-level-feature/data/'
root_path="/mnt/e/VAST/Skin_detect/skin_data"
folder_names=[name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
batch_size = 32
num_workers = 4
list_csv_test=[f'/mnt/e/VAST/Low-level-feature/data/{i}/vaynen_test_{i}.csv' for i in folder_names]
list_csv_train=[f'/mnt/e/VAST/Low-level-feature/data/{i}/vaynen_train_{i}.csv' for i in folder_names]
for train_csv,test_csv in tqdm(list(zip(list_csv_train,list_csv_test)),total=len(list_csv_train), desc="csv"):
    train_dataset = CSVImageMetadataDataset(csv_file=train_csv, root_dir=root_path)
    
    test_dataset = CSVImageMetadataDataset(csv_file=test_csv, root_dir=root_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    sample_batch = next(iter(train_dataloader))
    _, metadata_sample, _ = sample_batch  
    feature_size = metadata_sample.shape[1]
    print(f'training csv: {train_csv}')
    model=ResNetHybrid(resnet_type='resnet50', out_dim=3, n_meta_features=feature_size,pretrained=True)
    train_model(model,train_loader=train_dataloader,epochs=30,features_set= train_csv)
    result=evaluate_model(model,test_dataloader,train_csv)

    with open("./hybrid_model/results.txt", "a") as f:
        f.write(f"{result}\n")
    
    torch.save(model, f'./hybrid_model/model_{train_csv}.pth')