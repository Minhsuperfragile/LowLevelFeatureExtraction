import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_yolo import YOLOHybrid  # Import YOLOHybrid từ models.py
from utils.dataset import *
import torch.optim as optim
import torch.nn as nn

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(target_size=224):
    """
    Transform pipeline cho ảnh:
      - Resize, center crop, chuyển tensor và normalize theo ImageNet.
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

def train_model(model, train_loader, epochs: int, features_set: str, loss_path='loss_history.npy'):
    model.to(device)
    model.train()
    print(f"Training model using {device}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for images, metadata, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.long().to(device)
            
            # Forward pass qua YOLO
            outputs = model(images, metadata)
            # Tính loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    np.save(loss_path, np.array(loss_history))
    print(f"Training on {features_set} complete!")

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, metadata, labels in test_loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            outputs = model(images, metadata)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels)
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Cấu hình dữ liệu
path = r'C:\Vast\Skin_detect\vaynen_data'
root_path = r'C:\Users\Acer\skindata\skin_data'
batch_size = 32
num_workers = 4
csv_test = r'C:\Vast\Skin_detect\vaynen_data\vip\vaynen_test_vip_new.csv' 
csv_train = r'C:\Vast\Skin_detect\vaynen_data\vip\vaynen_train_vip_new.csv'

train_dataset = CSVImageMetadataDataset(csv_file=csv_train, root_dir=root_path)
test_dataset = CSVImageMetadataDataset(csv_file=csv_test, root_dir=root_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Lấy số lượng đặc trưng metadata
sample_batch = next(iter(train_dataloader))
_, metadata_sample, _ = sample_batch  
feature_size = metadata_sample.shape[1]
print(f'Training CSV: {csv_train}')

# Khởi tạo mô hình YOLOHybrid
yolo_model_path = "c:/Vast/Skin_detect/models_2classVN_v2_Yolov8_20250216_001515.pt"
model = YOLOHybrid(yolo_model_path=yolo_model_path, out_dim=2, n_meta_features=feature_size)

# Huấn luyện và đánh giá mô hình
train_model(model, train_loader=train_dataloader, epochs=1, features_set='vip')
result = evaluate_model(model, test_dataloader)

with open("results.txt", "a") as f:
    f.write(f"{result}\n")

torch.save(model, 'model_vip_yolo.pth')
