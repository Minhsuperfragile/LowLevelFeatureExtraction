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

list_of_folder = os.listdir("data/")

for llf_param_set in param_list[:1]:
    llf = LowLevelFeatureExtractor(**llf_param_set)

    llf_name = llf.function.__name__

    model = SimpleNeuralNetwork(inputs= llf.get_features_size())

    train_model(model, train_dataloader, llf, 1, llf_name)
    result = evaluate_model(model, test_dataloader, llf, llf_name)

    with open("./ckpts/results.txt", "a") as f:
        f.write(f"{result}\n")

    torch.save(model.state_dict(), f'./ckpts/model_{llf_name}.pth')