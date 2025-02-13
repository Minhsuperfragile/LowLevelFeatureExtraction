from torchvision import transforms
from torch.utils.data import DataLoader
import os
from utils import *

ckpt_folder = "./ckpts/"
# Check if the directory already exists
if not os.path.exists(ckpt_folder):
    # Create the directory using mkdir()
    os.makedirs(ckpt_folder)
    print(f"Folder '{ckpt_folder}' created.")

# root_folder = "../skin_data/"
root_folder = "/mnt/c/Users/trong/Documents/skin_data" 

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale,
    transforms.ToTensor()  # Convert to tensor
])

train_dataset = CSVImageMetadataDataset(csv_file='./data/vaynen_train_linux.csv', root_dir=root_folder, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)   

test_dataset = CSVImageMetadataDataset(csv_file='./data/vaynen_test_linux.csv', root_dir=root_folder, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

for llf_param_set in param_list[:1]:
    llf = LowLevelFeatureExtractor(**llf_param_set)

    llf_name = llf.function.__name__

    model = SimpleNeuralNetwork(inputs= llf.get_features_size())

    train_model(model, train_dataloader, llf, 3, llf_name)
    result = evaluate_model(model, test_dataloader, llf, llf_name)

    with open("./ckpts/results.txt", "a") as f:
        f.write(f"{result}\n")

    torch.save(model.state_dict(), f'./ckpts/model_{llf_name}.pth')